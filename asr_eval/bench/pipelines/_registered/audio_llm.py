from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.base.longform import ContextualLongformVAD, LongformVAD
from asr_eval.models.pyannote_vad import PyannoteSegmenter
from asr_eval.models.vikhr_wrapper import VikhrBorealisWrapper
from asr_eval.models.voxtral_wrapper import VoxtralWrapper
from asr_eval.models.gemma_wrapper import Gemma3nWrapper
from asr_eval.models.qwen2_audio_wrapper import Qwen2AudioWrapper
from asr_eval.models.qwen3_asr_wrapper import (
    ORUK_FOURIER_2B_MODEL_ID,
    ORUK_FOURIER_2B_REVISION,
    Qwen3ASRWrapper,
)
from asr_eval.models.flamingo_wrapper import FlamingoWrapper


# Vikhr Borealis

class _(TranscriberPipeline, register_as='vikhr-borealis-vad'):
    def init(self):
        return LongformVAD(VikhrBorealisWrapper(), PyannoteSegmenter())


# Flamingo

class _(TranscriberPipeline, register_as='flamingo-ru-vad'):
    def init(self):
        return LongformVAD(FlamingoWrapper(lang='ru'), PyannoteSegmenter())


# Gemma3n

class _(TranscriberPipeline, register_as='gemma3n-vad'):
    def init(self):
        return LongformVAD(Gemma3nWrapper(), PyannoteSegmenter())

class _(TranscriberPipeline, register_as='gemma3n-ru-vad'):
    def init(self):
        return LongformVAD(Gemma3nWrapper(lang='ru'), PyannoteSegmenter())


class _(TranscriberPipeline, register_as='gemma3n-ru-vad-contextual'):
    def init(self):
        return ContextualLongformVAD(
            Gemma3nWrapper(lang='ru'),
            PyannoteSegmenter(),
            max_history_words=100,
        )

# Qwen2

class _(TranscriberPipeline, register_as='qwen2-audio-vad'):
    def init(self):
        return LongformVAD(Qwen2AudioWrapper(), PyannoteSegmenter())


# Qwen3-ASR

class _(TranscriberPipeline, register_as='qwen3-asr-1.7b-vad'):
    def init(self):
        return LongformVAD(
            Qwen3ASRWrapper('Qwen/Qwen3-ASR-1.7B-hf'),
            PyannoteSegmenter(),
        )


# Oruk Fourier 2B (optional English-only Qwen3-ASR model, CC BY-NC 4.0)
class _(TranscriberPipeline, register_as='oruk-fourier-2b-vad'):
    def init(self):
        return LongformVAD(
            Qwen3ASRWrapper(
                ORUK_FOURIER_2B_MODEL_ID,
                revision=ORUK_FOURIER_2B_REVISION,
                language='en',
            ),
            PyannoteSegmenter(),
        )

# Voxtral

class _(TranscriberPipeline, register_as='voxtral-3B'):
    def init(self):
        return VoxtralWrapper(
            'mistralai/Voxtral-Mini-3B-2507',
            language='ru',
            local_server_verbose=True,
        )


class _(TranscriberPipeline, register_as='voxtral-3B-mp3'):
    def init(self):
        return VoxtralWrapper(
            'mistralai/Voxtral-Mini-3B-2507',
            language='ru',
            local_server_verbose=True,
            format='mp3',
        )


class _(TranscriberPipeline, register_as='voxtral-24B'):
    def init(self):
        return VoxtralWrapper(
            'mistralai/Voxtral-Small-24B-2507',
            language='ru',
            local_server_verbose=True,
        )


class _(TranscriberPipeline, register_as='voxtral-24B-mp3'):
    def init(self):
        return VoxtralWrapper(
            'mistralai/Voxtral-Small-24B-2507',
            language='ru',
            local_server_verbose=True,
            format='mp3',
        )