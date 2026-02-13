from datasets import Audio, Dataset, Features, Value  # type: ignore

from asr_eval.bench.datasets._registry import DATASETS_DIR, register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids


@register_dataset('yandex-commands-train')
def load_yandex_commands(split: str = 'test') -> Dataset:
    '''
    Open ASR data: one-word Russian commands. Has also an unlabeled test set.
    
    Downloaded from: https://alphacephei.com/test-other/yacmd/yandex-cup-asr-data.tar.gz
    Provided by: Nikolai V. Shmyrev
    '''
    # In yandex-cup-asr-data, there is a labeled train split and unlabeled
    # (private) test split We only have a labeling for train split, and call
    # in "test" split in asr_eval'
    
    wav_paths = list(
        (DATASETS_DIR / 'yandex-cup-asr-data/speech_commands_train')
        .glob('*/*.wav')
    )

    return Dataset.from_dict( # type: ignore
        mapping={
            'audio': [str(p) for p in wav_paths],
            'transcription': [p.parent.name for p in wav_paths],
        },
        features=Features({
            'audio': Audio(sampling_rate=16000),
            'transcription': Value("string"),
        })
    ).map(assign_sample_ids, with_indices=True)