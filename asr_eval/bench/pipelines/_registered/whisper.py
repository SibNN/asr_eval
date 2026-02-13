from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.whisper_faster_wrapper import FasterWhisperLongformWrapper
from asr_eval.models.whisper_wrapper import WhisperLongformWrapper
from asr_eval.streaming.wrappers import OfflineToStreaming


# Whisper base checkpoints

class _(TranscriberPipeline, register_as='whisper-tiny'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-tiny')

class _(TranscriberPipeline, register_as='whisper-small'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-small')

class _(TranscriberPipeline, register_as='whisper-medium'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-medium')

class _(TranscriberPipeline, register_as='whisper-large-v3'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-large-v3')
    
# Whisper turbo

class _(TranscriberPipeline, register_as='whisper-large-v3-turbo'):
    def init(self):
        return WhisperLongformWrapper('openai/whisper-large-v3-turbo')

# Whisper fine-tuned

class _(TranscriberPipeline, register_as='whisper-podlodka-turbo'):
    def init(self):
        return WhisperLongformWrapper('bond005/whisper-podlodka-turbo')

# Faster-Whisper

class _(TranscriberPipeline, register_as='faster-whisper-internal-vad'):
    def init(self):
        return FasterWhisperLongformWrapper(
            segmenter='internal', dtype='float32'
        )

# got segmentation fault
# https://github.com/SYSTRAN/faster-whisper/issues/1371
# class _(TranscriberPipeline, register_as='faster-whisper-pyannote-vad'):
#     def init(self):
#         return FasterWhisperLongformWrapper(
#             segmenter=PyannoteSegmenter(), dtype='float32'
#         )

# Quasi-streaming

_name = 'whisper-podlodka-turbo-quasi-streaming'
class _(TranscriberPipeline, register_as=_name):
    def init(self):
        return OfflineToStreaming(
            WhisperLongformWrapper('bond005/whisper-podlodka-turbo'),
            interval=1,
        )
