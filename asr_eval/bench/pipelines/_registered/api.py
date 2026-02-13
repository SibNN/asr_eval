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
            language='ru-RU',
            normalize=False,
        )
        
# Salute

class _(TranscriberPipeline, register_as='salute-api'):
    def init(self):
        return SaluteWrapper(api_key=os.environ['SALUTE_KEY'])