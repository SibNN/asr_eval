from __future__ import annotations
import os

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.yandex_speechkit_wrapper import YandexSpeechKitWrapper
from asr_eval.models.salute_wrapper import SaluteWrapper


# Yandex SpeechKit

class _(TranscriberPipeline, register_as='yandex-speechkit'):
    def init(self):
        return YandexSpeechKitWrapper(
            api_key=os.environ['YANDEX_API_KEY'],
            language='auto',
            normalize=False,
        )

class _(TranscriberPipeline, register_as='yandex-speechkit-ru'):
    def init(self):
        return YandexSpeechKitWrapper(
            api_key=os.environ['YANDEX_API_KEY'],
            language='ru-RU',
            normalize=False,
        )
        
# Salute

class _(TranscriberPipeline, register_as='salute-api-en'):
    def init(self):
        return SaluteWrapper(
            api_key=os.environ['SALUTE_KEY'], language='en-US'
        )

class _(TranscriberPipeline, register_as='salute-api-ru'):
    def init(self):
        return SaluteWrapper(
            api_key=os.environ['SALUTE_KEY'], language='ru-RU'
        )