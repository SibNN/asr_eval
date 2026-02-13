from datasets import Audio, load_dataset, Dataset  # type: ignore

from asr_eval.bench.datasets._registry import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('youtube-lectures')
def load_youtube_lectures(split: str = 'test') -> Dataset:
    # dangrebenkin/long_audio_youtube_lectures has a single split
    # called "test" in asr_eval and "train" on HF
    return (
        load_dataset('dangrebenkin/long_audio_youtube_lectures', split='train')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
    )
    # return (
    #     load_from_disk('/asr_datasets/long_audio_youtube_lectures')
    #     .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
    #     .map(_assign_sample_ids, with_indices=True)
    # )