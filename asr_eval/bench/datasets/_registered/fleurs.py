from datasets import Audio, load_dataset, Dataset # type: ignore

from asr_eval.bench.datasets._registry import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('fleurs-ru', splits=('train', 'validation', 'test'))
def load_fleurs(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            'google/fleurs',
            name='ru_ru',
            split=split,
            trust_remote_code=True,
        )
        .remove_columns('transcription')
        .rename_column('raw_transcription', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )