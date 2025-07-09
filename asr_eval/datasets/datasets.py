from typing import cast
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets # type: ignore


# A standardized set of ASR test datasets

def load_multivariant_v1_200() -> Dataset:
    return cast(Dataset, load_from_disk('/asr_datasets/multivariant_v1_200'))

def load_youtube_lectures() -> Dataset:
    # "train" is a single split here
    # loading dangrebenkin/long_audio_youtube_lectures from HF gives an error with datasets==3.6.0
    # https://github.com/huggingface/datasets/issues/7676
    # return cast(Dataset, load_dataset('dangrebenkin/long_audio_youtube_lectures', split='train'))
    return cast(Dataset, load_from_disk('/asr_datasets/long_audio_youtube_lectures'))

def load_golos_farfield() -> Dataset:
    return cast(Dataset, load_dataset('bond005/sberdevices_golos_100h_farfield', split='test'))

def load_rulibrispeech() -> Dataset:
    return cast(Dataset, load_dataset('bond005/rulibrispeech', split='test'))

def load_podlodka() -> Dataset:
    return cast(Dataset, load_dataset('bond005/podlodka_speech', split='test'))

def load_podlodka_full() -> Dataset:
    return concatenate_datasets([
        cast(Dataset, load_dataset('bond005/podlodka_speech', split='test')),
        cast(Dataset, load_dataset('bond005/podlodka_speech', split='train')),
        cast(Dataset, load_dataset('bond005/podlodka_speech', split='validation')),
    ])

def load_sova_rudevices() -> Dataset:
    return cast(Dataset, load_dataset('bond005/sova_rudevices', split='test'))

def load_resd() -> Dataset:
    return (
        cast(Dataset, load_dataset('Aniemore/resd_annotated', split='test'))
        .rename_column('text', 'transcription')
        .rename_column('speech', 'audio')
    )

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

def load_speech_massive() -> Dataset:
    return (
        cast(Dataset, load_dataset(
            'FBK-MT/Speech-MASSIVE-test',
            name='ru-RU',
            split='test',
        ))
        .rename_column('utt', 'transcription')
    )

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