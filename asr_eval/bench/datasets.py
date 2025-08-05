from typing import Callable, TypedDict, cast
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets # type: ignore

from ..utils.types import FLOATS # type: ignore


class AudioData(TypedDict):
    array: FLOATS
    sampling_rate: int


class AudioSample(TypedDict):
    audio: AudioData
    transcription: str


datasets_registry: dict[str, Callable[[], Dataset]]


def register_dataset(name: str):
    def decorator(fn: Callable[[], Dataset]):
        assert name not in datasets_registry
        datasets_registry[name] = fn
        return fn
    return decorator


@register_dataset('multivariant-v1-200')
def load_multivariant_v1_200() -> Dataset:
    return cast(Dataset, load_from_disk('/asr_datasets/multivariant_v1_200'))

@register_dataset('youtube-lectures')
def load_youtube_lectures() -> Dataset:
    # "train" is a single split here
    # loading dangrebenkin/long_audio_youtube_lectures from HF gives an error with datasets==3.6.0
    # https://github.com/huggingface/datasets/issues/7676
    # return cast(Dataset, load_dataset('dangrebenkin/long_audio_youtube_lectures', split='train'))
    return cast(Dataset, load_from_disk('/asr_datasets/long_audio_youtube_lectures'))

@register_dataset('golos-farfield')
def load_golos_farfield() -> Dataset:
    return cast(Dataset, load_dataset('bond005/sberdevices_golos_100h_farfield', split='test'))

@register_dataset('rulibrispeech')
def load_rulibrispeech() -> Dataset:
    return cast(Dataset, load_dataset('bond005/rulibrispeech', split='test'))

@register_dataset('podlodka')
def load_podlodka() -> Dataset:
    return cast(Dataset, load_dataset('bond005/podlodka_speech', split='test'))

@register_dataset('podlodka-full')
def load_podlodka_full() -> Dataset:
    return concatenate_datasets([
        cast(Dataset, load_dataset('bond005/podlodka_speech', split='test')),
        cast(Dataset, load_dataset('bond005/podlodka_speech', split='train')),
        cast(Dataset, load_dataset('bond005/podlodka_speech', split='validation')),
    ])

@register_dataset('sova-rudevices')
def load_sova_rudevices() -> Dataset:
    return cast(Dataset, load_dataset('bond005/sova_rudevices', split='test'))

@register_dataset('resd')
def load_resd() -> Dataset:
    return (
        cast(Dataset, load_dataset('Aniemore/resd_annotated', split='test'))
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
    )

@register_dataset('fleurs')
def load_fleurs() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'google/fleurs',
            name='ru_ru',
            split='test',
            trust_remote_code=True,
        ))
        .remove_columns('transcription')
        .rename_column('raw_transcription', 'transcription')
    )

@register_dataset('speech-massive')
def load_speech_massive() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'FBK-MT/Speech-MASSIVE-test',
            name='ru-RU',
            split='test',
        ))
        .rename_column('utt', 'transcription')
    )

@register_dataset('common-voice-17.0')
def load_common_voice_17_0() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'mozilla-foundation/common_voice_17_0',
            name='ru',
            split='test',
            trust_remote_code=True,
        ))
        .rename_column('sentence', 'transcription')
    )