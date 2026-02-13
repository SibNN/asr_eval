from __future__ import annotations
from collections.abc import Sequence
from functools import cache
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, TypedDict, cast

import pandas as pd

if TYPE_CHECKING:
    from datasets import Dataset

from asr_eval import CACHE_DIR, check_datasets_package, has_datasets_package
from asr_eval.utils.types import FLOATS
from asr_eval.bench.datasets.mappers import remove_samples


DATASETS_DIR = CACHE_DIR / 'datasets'
"""A directory where the non-HuggingFace datasets are stored.

By default, ~/asr_eval/datasets
"""


class AudioData(TypedDict):
    """A TypedDict typization for
    `Audio <https://huggingface.co/docs/datasets/en/about_dataset_features#audio-feature>`_
    feature in Hugging Face dataset.
    
    See examples in the docs for
    :class:`~asr_eval.bench.datasets.AudioSample`.
    """
    
    array: FLOATS
    """1-D audio waveform of floats, normalized roughly from -1 to 1,
    with sampling rate specified in
    :attr:`~asr_eval.bench.datasets.AudioData.sampling_rate` (normally
    16000).
    """
    
    sampling_rate: int
    """A sampling rate for
    :attr:`~asr_eval.bench.datasets.AudioData.array`, i. e. array size
    per second (normally 16000).
    """
    


class AudioSample(TypedDict):
    """A TypedDict typization for Hugging Face audio sample in
    standard asr_eval format.
    
    This class is for typing purposes only. A sample in Hugging Face
    dataset is a plain dict.
    
    In asr_eval standard workflow, sampling rate should be 16_000 and
    all samples should have unique "sample_id" value. Dataset may
    include other custom fields as well.
    
    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.
    
    Example:
    
        >>> # instantiation from `get_dataset`:
        >>> from asr_eval.bench.datasets import get_dataset
        >>> dataset = get_dataset('podlodka')
        >>> sample: AudioSample = dataset[0]
        
        >>> # AudioSample inner structure:
        >>> audio_data: AudioData = sample['audio']
        >>> assert audio_data['sampling_rate'] == 16_000
        >>> waveform: FLOATS = audio_data['array']
        >>> transcription: str = sample['transcription']
        
        >>> # instantiation from Hugging Face:
        >>> from datasets import load_dataset, Audio
        >>> from asr_eval.bench.datasets import AudioSample, AudioData
        >>> from asr_eval.utils.types import FLOATS
        >>> from asr_eval.bench.datasets.mappers import assign_sample_ids
        >>> dataset = (
        ...     load_dataset('PolyAI/minds14', name='en-US', split='train')
        ...     .cast_column('audio', Audio(sampling_rate=16_000))
        ...     .map(assign_sample_ids, with_indices=True)
        ... )
        >>> sample: AudioSample = dataset[0]
    """
    
    audio: AudioData
    """An `Audio <https://huggingface.co/docs/datasets/en/about_dataset_features#audio-feature>`_
    feature. In asr_eval standard workflow, should be obtained with
    :code:`.cast_column('audio', Audio(decode=True, sampling_rate=16_000))`.
    """
    
    transcription: str
    """A transcription as text, possibly with multivariant annotation,
    may optionally include punctuation or capitalization.
    """
    
    sample_id: int
    """A sample ID that should be unique in the dataset. Normally should
    equal a sample index in the unshuffled and not filtered version."""
    


@dataclass
class DatasetInfo:
    """A container for dataset information that is stored if a dataset
    gets registered.
    """
    instantiate_fn: Callable[[str], Dataset]
    splits: tuple[str, ...]
    unlabeled: bool
    
    # filters: split -> instantiate_fn
    filter: Callable[[str], list[int]] | None = None


datasets_registry: dict[str, DatasetInfo] = {}
"""A registry where all the registered dataset info is stored."""

def register_dataset(
    name: str,
    splits: tuple[str, ...] = ('test',),
    unlabeled: bool = False,
):
    """Register a new dataset in asr_eval. The dataset will be available
    under the registered name in
    :func:`~asr_eval.bench.datasets.get_dataset`.
    
    Args:
        name: A unique name for the dataset.
        splits: A list of available splits. All the datasets should at
            least have "test" split available, because asr_eval is for
            testing purposes. If a dataset has a "train" split only,
            consider registering it as "test" if you want to test on it.
            Datasets can have other splits registered with any names,
            primarily to check for train-test overlap.
        unlabeled: If the dataset is unlabeled. Experimental feature.
    
    See many examples in `asr_eval.bench.datasets._registered` package.
    
    Example:
        >>> from datasets import Audio, load_dataset, Dataset
        >>> from asr_eval.bench.datasets import register_dataset, get_dataset
        >>> from asr_eval.bench.datasets.mappers import assign_sample_ids
        >>> @register_dataset('podlodka-new', splits=('train', 'test'))
        >>> def load_podlodka(split: str = 'test') -> Dataset:
        ...     return (
        ...         load_dataset('bond005/podlodka_speech', split=split)
        ...         .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ...         .map(assign_sample_ids, with_indices=True)
        ...     )
        >>> dataset = get_dataset('podlodka-new')
    """
    global datasets_registry
    def decorator(fn: Callable[[str], Dataset]):
        assert name not in datasets_registry
        datasets_registry[name] = DatasetInfo(
            instantiate_fn=fn,
            unlabeled=unlabeled,
            splits=splits,
        )
        return fn
    return decorator


def set_filter(dataset_name: str):
    """Register a sample filter for the given registered dataset.
    
    The filter should accept split name and return a list of sample IDs
    to filter out. Is primarily used for deduplication. The
    :func:`~asr_eval.bench.datasets.get_dataset` function by default
    returns a filtered dataset if filter was set.
    """
    global datasets_registry
    def decorator(fn: Callable[[str], list[int]]):
        assert dataset_name in datasets_registry
        datasets_registry[dataset_name].filter = fn
        return fn
    return decorator


def load_relabeling_from_file(
    path: str | Path, encoding: str | None = None
) -> dict[int, str]:
    raw_text = Path(path).read_text(encoding=encoding)
    relabeling: dict[int, str] = {}
    for line in raw_text.splitlines():
        if not line.lstrip().startswith('#'):
            idx, transcription = line.lstrip().replace('\t', ' ').split(' ', 1)
            relabeling[int(idx)] = transcription
    return relabeling


@cache
def get_dataset(
    name: str,
    augmentor_name: str | None | Literal['none'] = None,
    split: str = 'test',
    shuffle: bool = True,
    filter: bool = True,
) -> Dataset:
    """Instantiates a registered dataset.
    
    Args:
        name: A dataset name under which it was registered.
        augmentor_name: An augmentor name to apply, None by default
            (see :class:`~asr_eval.bench.augmentors.AudioAugmentor`).
        split: A split name, "test" by default.
        shuffle: Whether to perform :code:`shuffle(seed=0)`, True by
            default. The shuffling is used to ensure that the first N
            samples form a representative set. The sample IDs help to
            track the original indices, before shuffling or filtering.
        filter: Whether to filter out duplicate and malformed samples,
            if :class:`~asr_eval.bench.datasets.set_filter` was done
            for this dataset. True by default. This ensures that the
            datasets in asr_eval by default do not contain duplicate
            of malformed samples.
    """
    check_datasets_package()
    
    # instantiating
    dataset_info = get_dataset_info(name)
    assert split in dataset_info.splits, (
        f'No split {split} for {name}, available splits: {dataset_info.splits}'
    )
    dataset = dataset_info.instantiate_fn(split)
    
    # processing
    if augmentor_name and augmentor_name != 'none':
        from asr_eval.bench.augmentors import get_augmentor
        augmentor = get_augmentor(augmentor_name)
        dataset = dataset.map(augmentor) # type: ignore
    
    # shuffling
    if shuffle:
        dataset = dataset.shuffle(0)
        
    # filtering
    if filter:
        sample_ids_to_remove = set(get_filter(name, split))
        if len(sample_ids_to_remove):
            # This transform, as usual, will be cached on disk. When
            # .filter(fn, fn_kwargs=kwargs) is called, both fn and kwargs
            # affect the resulting fingerprint. The fingerprint is calculated
            # by hashing the result of pickle.dumps (see
            # datasets.fingerprint.Hasher) it works also for sets, because
            # pickle.dumps is able to dump sets consistently
            dataset = dataset.filter( # type: ignore
                remove_samples,
                fn_kwargs={'sample_ids_to_remove': sample_ids_to_remove},
            )
    
    return dataset


@cache
def get_filter(dataset_name: str, split: str = 'test') -> list[int]:
    filter_fn = get_dataset_info(dataset_name).filter
    if filter_fn is None:
        return []
    return filter_fn(split)


@cache
def get_dataset_index(name: str) -> int:
    if not has_datasets_package:
        raise RuntimeError(
            'Datasets package not found, run `pip install datasets`.'
        )
    return list(datasets_registry).index(name)


def get_dataset_info(name: str) -> DatasetInfo:
    """Get info for a registered dataset."""
    check_datasets_package()
    if name not in datasets_registry:
        raise ValueError(f'Dataset was not registered in asr_eval: {name}')
    return datasets_registry[name]


_datasets_cache: dict[
    tuple[str, str],  # dataset_name, split
    tuple[Dataset, dict[int, int]]  # dataset, id mapping
] = {}

def get_dataset_sample_by_id(
    dataset_name: str,
    split: str,
    sample_id: int,
    augmentor_name: str | None = None,
) -> AudioSample:
    """An utility to simply retrieve the required sample ID for the
    given dataset. Internally instantiates a dataset if not instantiated
    yet.
    """
    if (dataset_name, split) not in _datasets_cache:
        dataset = get_dataset(
            dataset_name,
            split=split,
            filter=False,
            shuffle=False,
            augmentor_name=augmentor_name,
        )
        mapping = {
            sample['sample_id']: pos
            for pos, sample in enumerate(cast(Sequence[AudioSample], dataset))
        }
        _datasets_cache[dataset_name, split] = (dataset, mapping)
    dataset, mapping = _datasets_cache[dataset_name, split]
    if sample_id not in mapping:
        raise ValueError(
            f'Sample id {sample_id} not found'
            f' for dataset {dataset_name}, split {split}'
        )
    pos = mapping[sample_id]
    return cast(AudioSample, dataset[pos])


def _add_custom_annotations_from_csv(path: str | Path): # pyright: ignore[reportUnusedFunction]
    global _custom_annotations

    df = pd.read_csv(path) # pyright: ignore[reportUnknownMemberType]
    columns = ['dataset_name', 'sample_id', 'text']
    for col_name in columns:
        assert col_name in df.columns, (
            f'Column {col_name} not found in the'
            ' `custom_annotations` file'
        )
    for _, row in df.iterrows():
        _custom_annotations[row['dataset_name'], row['sample_id']] = (  # type: ignore
            row['text']
        )

_custom_annotations: dict[tuple[str, int], str] = {}

def _get_transcription_by_sample_id( # pyright: ignore[reportUnusedFunction]
    dataset_name: str,
    sample_id: int,
):
    # similar to `get_dataset_sample_by_id`, but supports also
    # custom annotations for datasets not registered in asr_eval, for which
    # annotations are directly provided with _add_custom_annotations
    if text := _custom_annotations.get(
        (dataset_name, sample_id), None
    ):
        return text
    
    sample = get_dataset_sample_by_id(
        dataset_name, split='test', sample_id=sample_id
    )
    assert sample['transcription'] is not None, (
        f'transcription is None for {dataset_name=} and {sample_id=}'
    )
    return sample['transcription']