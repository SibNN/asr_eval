from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.base.longform import LongformVAD
from asr_eval.models.pyannote_vad import PyannoteSegmenter
from asr_eval.models.vosk54_wrapper import VoskV54
from asr_eval.models.vosk_streaming_wrapper import VoskStreaming


class _(TranscriberPipeline, register_as='vosk-0.54-vad'):
    def init(self):
        return LongformVAD(VoskV54(), PyannoteSegmenter())
    
class _(TranscriberPipeline, register_as='vosk-ru-0.42-streaming'):
    def init(self):
        return VoskStreaming('vosk-model-ru-0.42', chunk_length_sec=1)