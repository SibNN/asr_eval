from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Callable, TypedDict
from datasets import Audio, load_dataset, load_from_disk, Dataset, concatenate_datasets # type: ignore

from asr_eval import ROOT_DIR # type: ignore
from ..utils.types import FLOATS # type: ignore


__all__ = [
    'AudioData',
    'AudioSample',
    'datasets_registry',
    'get_dataset',
    'register_dataset',
]


class AudioData(TypedDict):
    '''A typization for 'audio' field in HF datasets'''
    array: FLOATS
    sampling_rate: int


class AudioSample(TypedDict):
    '''A typization for audio-text samples in HF datasets'''
    audio: AudioData
    transcription: str
    

RELABELING_TYPE = dict[int, str]


@dataclass
class DatasetInfo:
    '''Info for a registered ASR dataset.'''
    instantiate_fn: Callable[[], Dataset]
    unlabeled: bool
    relabelings: dict[str, Callable[[], RELABELING_TYPE]] = field(default_factory=dict)


datasets_registry: dict[str, DatasetInfo] = {}


@cache
def get_dataset(name: str) -> Dataset:
    '''Get a registered ASR dataset. See the examples in the current file.'''
    return get_dataset_info(name).instantiate_fn()


@cache
def get_dataset_index(name: str) -> int:
    '''Get an index (in registration order) for a registered ASR dataset.'''
    return list(datasets_registry).index(name)


def get_dataset_info(name: str) -> DatasetInfo:
    '''Get info for a registered ASR dataset.'''
    if name not in datasets_registry:
        raise ValueError(f'Dataset does not exist: {name}')
    return datasets_registry[name]


def register_dataset(name: str, unlabeled: bool = False):
    '''
    Register a new ASR dataset. See the examples in the current file.
    '''
    global datasets_registry
    def decorator(fn: Callable[[], Dataset]):
        assert name not in datasets_registry
        datasets_registry[name] = DatasetInfo(instantiate_fn=fn, unlabeled=unlabeled)
        return fn
    return decorator


def register_relabeling(dataset_name: str, name: str):
    '''
    Register a new relabeling for a registered ASR dataset. See the examples in the current file.
    '''
    global datasets_registry
    def decorator(fn: Callable[[], RELABELING_TYPE]):
        assert dataset_name in datasets_registry
        assert name not in datasets_registry[dataset_name].relabelings
        datasets_registry[dataset_name].relabelings[name] = fn
        return fn
    return decorator

def load_relabeling_from_file(path: str | Path) -> RELABELING_TYPE:
    raw_text = Path(path).read_text()
    relabeling: RELABELING_TYPE = {}
    for line in raw_text.splitlines():
        if not line.lstrip().startswith('#'):
            idx, transcription = line.lstrip().split(' ', 1)
            relabeling[int(idx)] = transcription
    return relabeling


@register_dataset('multivariant-v1-200')
def load_multivariant_v1_200() -> Dataset:
    return (
        load_from_disk('/asr_datasets/multivariant_v1_200')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('common-voice-17.0')
def load_common_voice_17_0() -> Dataset:
    return (
        load_dataset(
            'mozilla-foundation/common_voice_17_0',
            name='ru',
            split='test',
            trust_remote_code=True,
        )
        .rename_column('sentence', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('golos-farfield')
def load_golos_farfield() -> Dataset:
    return (
        load_dataset('bond005/sberdevices_golos_100h_farfield', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_relabeling('golos-farfield', 'multivariant')
def load_golos_farfield_multivariant() -> RELABELING_TYPE:
    return load_relabeling_from_file(ROOT_DIR / 'datasets/relabelings/golos-farfield.txt')

@register_dataset('rulibrispeech')
def load_rulibrispeech() -> Dataset:
    return (
        load_dataset('bond005/rulibrispeech', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('podlodka')
def load_podlodka() -> Dataset:
    return (
        load_dataset('bond005/podlodka_speech', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('podlodka-full')
def load_podlodka_full() -> Dataset:
    return concatenate_datasets([
        (
            load_dataset('bond005/podlodka_speech', split='test')
            .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ),
        (
            load_dataset('bond005/podlodka_speech', split='train')
            .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ),
        (
            load_dataset('bond005/podlodka_speech', split='validation')
            .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        ),
    ]).shuffle(0)

@register_dataset('sova-rudevices')
def load_sova_rudevices() -> Dataset:
    return (
        load_dataset('bond005/sova_rudevices', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('resd')
def load_resd() -> Dataset:
    return (
        load_dataset('Aniemore/resd_annotated', split='test')
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('fleurs')
def load_fleurs() -> Dataset:
    return (
        load_dataset(
            'google/fleurs',
            name='ru_ru',
            split='test',
            trust_remote_code=True,
        )
        .remove_columns('transcription')
        .rename_column('raw_transcription', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('speech-massive')
def load_speech_massive() -> Dataset:
    return (
        load_dataset(
            'FBK-MT/Speech-MASSIVE-test',
            name='ru-RU',
            split='test',
        )
        .rename_column('utt', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('youtube-lectures')
def load_youtube_lectures() -> Dataset:
    # "train" is a single split here
    # loading dangrebenkin/long_audio_youtube_lectures from HF gives an error with datasets==3.6.0
    # https://github.com/huggingface/datasets/issues/7676
    # return cast(Dataset, load_dataset('dangrebenkin/long_audio_youtube_lectures', split='train'))
    return (
        load_from_disk('/asr_datasets/long_audio_youtube_lectures')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )

@register_dataset('ontico-unlabeled', unlabeled=True)
def load_ontico_unlabeled() -> Dataset:
    return (
        load_from_disk('/asr_datasets/ontico_unlabeled')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .shuffle(0)
    )