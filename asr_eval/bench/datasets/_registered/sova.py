import json
from typing import cast

from datasets import Audio, load_dataset, Dataset  # type: ignore

from asr_eval import ROOT_DIR
from asr_eval.bench.datasets._registry import (
    load_relabeling_from_file, register_dataset, set_filter
)
from asr_eval.bench.datasets.mappers import (
    assign_sample_ids, keep_samples, replace_transcriptions
)


# Sova Ru Devices dataset

@register_dataset('sova-rudevices', splits=('train', 'validation', 'test'))
def load_sova_rudevices(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            'bond005/sova_rudevices',
            split=split,
            # skip downloading large train part if we need only test part
            data_files={split: f'data/{split}*'},
            verification_mode='no_checks',
        )
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )

# Sova Ru Devices multivariant version (and single-variant counterpart with
# the same samples set)

@register_dataset('sova-rudevices-multivariant', splits=('test',))
def load_sova_rudevices_multivariant(split: str = 'test') -> Dataset:
    relabeling = load_relabeling_from_file(
        ROOT_DIR / '_data/relabelings/sova-rudevices.txt'
    )
    return (
        cast(Dataset, load_sova_rudevices(split=split)) # type: ignore
        .filter(
            keep_samples,
            fn_kwargs={'sample_ids_to_keep': set(relabeling)},
        )
        .map(replace_transcriptions, fn_kwargs={'replacements': relabeling})
    )

@register_dataset('sova-rudevices-single-variant', splits=('test',))
def load_sova_rudevices_single_variant(split: str = 'test') -> Dataset:
    # a version of sova-rudevices that contains only the sample ids from
    # sova-rudevices-multivariant to enable side by side comparison
    relabeling = load_relabeling_from_file(
        ROOT_DIR / '_data/relabelings/sova-rudevices.txt'
    )
    return (
        cast(Dataset, load_sova_rudevices(split=split)) # type: ignore
        .filter(
            keep_samples,
            fn_kwargs={'sample_ids_to_keep': set(relabeling)},
        )
    )

@set_filter('sova-rudevices')
@set_filter('sova-rudevices-multivariant')
@set_filter('sova-rudevices-single-variant')
def filter_sova_rudevices(split: str = 'test') -> list[int]:
    # the multivariant relabeling file does not contain any indices from
    # duplicates/sova-rudevices.json, so, we don't need to @set_filter for
    # sova-rudevices-multivariant; however, we still add it for
    # consistency and in case such indexes appear
    data = (ROOT_DIR / '_data/duplicates/sova-rudevices.json').read_text()
    return json.loads(data).get(split, [])

# Sova Ru Devices Audiobooks

@register_dataset('sova-rudevices-audiobooks', splits=('train', 'test'))
def load_sova_rudevices_audiobooks(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            'dangrebenkin/sova_rudevices_audiobooks',
            split=split,
            # skip downloading large train part if we need only test part
            data_files={split: f'data/{split}*'},
            verification_mode='no_checks',
        )
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )

@set_filter('sova-rudevices-audiobooks')
def filter_sova_rudevices_audiobooks(split: str = 'test') -> list[int]:
    data = (
        (ROOT_DIR / '_data/duplicates/sova-rudevices-audiobooks.json')
        .read_text()
    )
    return json.loads(data).get(split, [])