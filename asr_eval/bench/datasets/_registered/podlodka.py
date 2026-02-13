from datasets import Audio, load_dataset, Dataset, concatenate_datasets  # type: ignore

from asr_eval.bench.datasets._registry import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('podlodka', splits=('train', 'test'))
def load_podlodka(split: str = 'test') -> Dataset:
    # In bond005/podlodka_speech, train and test splits are avaliable.
    # Validation split is a copy of test split.
    return (
        load_dataset('bond005/podlodka_speech', split=split)
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )

@register_dataset('podlodka-full')
def load_podlodka_full(split: str = 'test') -> Dataset:
    return concatenate_datasets([
        (
            load_dataset('bond005/podlodka_speech', split='test')
            .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ),
        (
            load_dataset('bond005/podlodka_speech', split='train')
            .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ),
    ]).map(assign_sample_ids, with_indices=True)