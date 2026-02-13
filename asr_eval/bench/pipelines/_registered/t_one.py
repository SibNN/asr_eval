from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.base.longform import LongformVAD
from asr_eval.models.pyannote_vad import PyannoteSegmenter
from asr_eval.models.t_one_wrapper import TOneStreaming, TOneWrapper


class _(TranscriberPipeline, register_as='t-one-vad'):
    def init(self):
        return LongformVAD(TOneWrapper(), PyannoteSegmenter())


class _(TranscriberPipeline, register_as='t-one-streaming'):
    def init(self):
        return TOneStreaming()