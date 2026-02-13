from datasets import Audio, load_dataset, Dataset  # type: ignore

from asr_eval.bench.datasets._registry import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('rulibrispeech', splits=('train', 'validation', 'test'))
def load_rulibrispeech(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            'bond005/rulibrispeech',
            split=split,
            # skip downloading large train part if we need only test part
            data_files={split: f'data/{split}*'},
            verification_mode='no_checks',
        )
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )