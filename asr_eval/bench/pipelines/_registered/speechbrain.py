from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.speechbrain_wrapper import SpeechbrainStreaming


_name = 'speechbrain-conformer-gigaspeech-streaming'
class _(TranscriberPipeline, register_as=_name):
    def init(self):
        return SpeechbrainStreaming(
            model_name='speechbrain/asr-streaming-conformer-gigaspeech'
        )