from datasets import Audio, load_from_disk, Dataset, Features, Value # type: ignore
import srt

from asr_eval.bench.datasets._registry import DATASETS_DIR, register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('multivariant-v1-200')
def load_multivariant_v1_200(split: str = 'test') -> Dataset:
    return (
        load_from_disk(DATASETS_DIR / 'multivariant_v1_200')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
        # assign indices after shuffle here
        .map(assign_sample_ids, with_indices=True)
    )


@register_dataset('multivariant-v2')
def load_multivariant_v2(split: str = 'test') -> Dataset:
    wav_paths = sorted((DATASETS_DIR / 'multivariant_v2').glob('*.wav'))
    srts = [
        list(srt.parse(p.with_suffix('.srt').read_text())) for p in wav_paths  # type: ignore
    ]

    return Dataset.from_dict( # type: ignore
        mapping={
            'audio': [str(p) for p in wav_paths],
            'transcription': [
                ' '.join(x.content for x in lst) for lst in srts
            ],
        },
        features=Features({
            'audio': Audio(sampling_rate=16000),
            'transcription': Value("string"),
        })
    ).map(assign_sample_ids, with_indices=True)