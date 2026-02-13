import json
from typing import cast

from datasets import Audio, load_dataset, Dataset  # type: ignore

from asr_eval import ROOT_DIR
from asr_eval.bench.datasets._registry import (
    load_relabeling_from_file, register_dataset, set_filter
)
from asr_eval.bench.datasets.mappers import (
    assign_sample_ids,
    keep_samples,
    remove_samples_with_none_transcription,
    replace_transcriptions,
)


@register_dataset('golos-farfield', splits=('train', 'validation', 'test'))
def load_golos_farfield(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            'bond005/sberdevices_golos_100h_farfield',
            split=split,
            # skip downloading large train part if we need only test part
            data_files={split: f'data/{split}*'},
            verification_mode='no_checks',
        )
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
        .filter(remove_samples_with_none_transcription)
    )

@register_dataset('golos-farfield-multivariant', splits=('test',))
def load_golos_farfield_multivariant(split: str = 'test') -> Dataset:
    relabeling = load_relabeling_from_file(
        ROOT_DIR / '_data/relabelings/golos-farfield.txt'
    )
    return (
        cast(Dataset, load_golos_farfield(split=split)) # type: ignore
        .filter(
            keep_samples,
            fn_kwargs={'sample_ids_to_keep': set(relabeling)},
        )
        .map(replace_transcriptions, fn_kwargs={'replacements': relabeling})
    )
    
@register_dataset('golos-crowd', splits=('train', 'validation', 'test'))
def load_golos_crowd(split: str = 'test') -> Dataset:
    # for some samples `sample['transcription']` is None, but the audios
    # contain speech; this looks like a bug in the dataset, we filter out these
    # samples regardless of the `filter` value in `get_dataset`, because such
    # samples are not valid AudioSample; we assign sample ids before this step,
    # this ensures that sample ids are indices in the original HF dataset, as
    # usual
    return (
        load_dataset(
            'bond005/sberdevices_golos_10h_crowd',
            split=split,
            # skip downloading large train part if we need only test part
            data_files={split: f'data/{split}*'},
            verification_mode='no_checks',
        )
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
        .filter(remove_samples_with_none_transcription)
    )

@set_filter('golos-crowd')
def filter_golos_crowd(split: str = 'test') -> list[int]:
    data = (ROOT_DIR / '_data/duplicates/golos-crowd.json').read_text()
    return json.loads(data).get(split, [])