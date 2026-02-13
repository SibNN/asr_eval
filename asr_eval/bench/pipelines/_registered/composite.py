from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.bench.pipelines._registered.gigaam import VOSK_LM
from asr_eval.correction.comparator_wordfreq import RuWordFreqComparator
from asr_eval.ctc.lm import CTCDecoderWithLM
from asr_eval.models.base.longform import LongformCTC
from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
from asr_eval.models.whisper_wrapper import WhisperLongformWrapper


class _(TranscriberPipeline, register_as='gigaam2-plus-whisper-wordfreq'):
    def init(self):
        return RuWordFreqComparator(
            CTCDecoderWithLM(
                LongformCTC(GigaAMShortformCTC('v2')),
                VOSK_LM,
            ),
            WhisperLongformWrapper('bond005/whisper-podlodka-turbo'),
        )