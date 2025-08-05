from typing import Callable, TypedDict
from datasets import Audio, load_dataset, load_from_disk, Dataset, concatenate_datasets # type: ignore

from ..utils.types import FLOATS # type: ignore


class AudioData(TypedDict):
    array: FLOATS
    sampling_rate: int


class AudioSample(TypedDict):
    audio: AudioData
    transcription: str


datasets_registry: dict[str, Callable[[], Dataset]] = {}


def register_dataset(name: str):
    global datasets_registry
    def decorator(fn: Callable[[], Dataset]):
        assert name not in datasets_registry
        datasets_registry[name] = fn
        return fn
    return decorator


@register_dataset('multivariant-v1-200')
def load_multivariant_v1_200() -> Dataset:
    return (
        load_from_disk('/asr_datasets/multivariant_v1_200')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
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
    )

@register_dataset('golos-farfield')
def load_golos_farfield() -> Dataset:
    return (
        load_dataset('bond005/sberdevices_golos_100h_farfield', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
    )

@register_dataset('rulibrispeech')
def load_rulibrispeech() -> Dataset:
    return (
        load_dataset('bond005/rulibrispeech', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
    )

@register_dataset('podlodka')
def load_podlodka() -> Dataset:
    return (
        load_dataset('bond005/podlodka_speech', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
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
    ])

@register_dataset('sova-rudevices')
def load_sova_rudevices() -> Dataset:
    return (
        load_dataset('bond005/sova_rudevices', split='test')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
    )

@register_dataset('resd')
def load_resd() -> Dataset:
    return (
        load_dataset('Aniemore/resd_annotated', split='test')
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
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
    )