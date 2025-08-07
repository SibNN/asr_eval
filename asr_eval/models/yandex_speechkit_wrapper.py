from typing import Literal, override

# Yandex SpeechKit imports
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

from ..segments.segment import TimedText
from .base.interfaces import TimedTranscriber
from ..utils.audio_ops import waveform_to_pydub
from ..utils.types import FLOATS


__all__ = [
    'YandexSpeechKitWrapper',
]


class YandexSpeechKitWrapper(TimedTranscriber):
    """
    A wrapper for Yandex SpeechKit transcriber.
    
    Docs:
    https://yandex.cloud/ru/docs/speechkit/stt/models
    
    To obtain API key, create service account and API key, as described:
    https://yandex.cloud/ru/docs/speechkit/quickstart/stt-quickstart-v2
    
    Speechkit provides timings for each word, raw and normalized text, It
    seems to normalize text for language='ru-Ru' but not for language='auto'.
    
    Example raw: 
    [седьмого [0.399, 1.060], восьмого [1.120, 1.780], мая [1.860, 2.399], в [2.520, 2.580],
     пуэрто [2.639, 3.340], рико [3.419, 3.899], прошел [4.110, 4.680], шестнадцатый [4.839, 5.839],
     этап [5.890, 6.299], формулы [6.470, 7.170], один [7.259, 7.740], с [7.859, 7.890],
     фондом [8.040, 8.780], сто [8.950, 9.320], тысяч [9.429, 9.690], долларов [9.900, 10.700],
     победителем [11.559, 12.346], стал [12.420, 12.733],
    
    Example normalized:
    7 8 Мая в Пуэрто Рико прошел 16 этап Формулы 1 с Фондом 10.00000000000% $-победителем стал
    
    As you can see, normalization introduces some errors, and it is sometimes hard to align raw and
    normalized text.
    
    If normalize=True and normalized text is returned by the API:
    1. transcribe() returns a full normalized text
    2. timed_transcribe() returns a list of normalized utterances if available, otherwise a fill text
    
    Otherwise:
    1. transcribe() returns a full unnormalized text
    2. timed_transcribe() returns a list of unnormalized single words
    
    Authors: Dmitry Ezhov & Oleg Sedukhin
    """
    def __init__(
        self,
        api_key: str,
        model: Literal['general', 'general:rc', 'general:deprecated'] = 'general',
        language: Literal['auto', 'ru-RU', 'en-US'] | str = 'ru-RU',
        audio_processing: AudioProcessingType = AudioProcessingType.Full,
        normalize: bool = False,
    ):
        configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=api_key))

        self.model = model_repository.recognition_model()
        self.model.model = model
        self.model.language = language
        self.model.audio_processing_type = audio_processing
        self.normalize = normalize

    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        results = self.model.transcribe(waveform_to_pydub(waveform))
        
        # We assume single-channel audio, so we take the first result
        result = results[0]
        
        outputs: list[TimedText] = []
        
        if self.normalize and result.normalized_text is not None:
            if result.utterances is not None:
                # return a list of normalized utterances
                for x in result.utterances:
                    assert x.normalized_text is not None
                    outputs.append(TimedText(
                        x.start_time_ms / 1000, x.end_time_ms / 1000, x.normalized_text
                    ))
            else:
                # return a full normalized text
                outputs.append(TimedText(
                    0, len(waveform) / 16_000, result.normalized_text
                ))
        else:
            # return a list of unnormalized words
            assert result.words is not None
            for x in result.words:
                outputs.append(TimedText(
                    x.start_time_ms / 1000, x.end_time_ms / 1000, x.word
                ))
        
        return outputs