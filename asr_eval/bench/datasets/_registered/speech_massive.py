from datasets import Audio, load_dataset, Dataset  # type: ignore

from asr_eval.bench.datasets._registry import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset(
    'speech-massive-ru', splits=('train_115', 'validation', 'test')
)
def load_speech_massive(split: str = 'test') -> Dataset:
    return (
        load_dataset(
            (
                'FBK-MT/Speech-MASSIVE-test'
                if split == 'test'
                else 'FBK-MT/Speech-MASSIVE'
            ),
            name='ru-RU',
            split=split,
        )
        .rename_column('utt', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )