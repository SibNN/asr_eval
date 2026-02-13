from __future__ import annotations

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.models.base.longform import LongformVAD
from asr_eval.models.nemo_wrapper import NvidiaNemoWrapper
from asr_eval.models.pyannote_vad import PyannoteSegmenter
from asr_eval.streaming.wrappers import OfflineToStreaming


class _(TranscriberPipeline, register_as='canary-1b-v2-vad'):
    def init(self):
        return LongformVAD(
            NvidiaNemoWrapper(
                'nvidia/canary-1b-v2',
                inference_kwargs={'source_lang': 'ru', 'target_lang': 'ru'},
                beam_size=16,
            ),
            PyannoteSegmenter(strict_limit_duration=30)
        )
    
class _(TranscriberPipeline, register_as='parakeet-tdt-0.6b-v3-vad'):
    def init(self):
        return LongformVAD(
            NvidiaNemoWrapper('nvidia/parakeet-tdt-0.6b-v3', beam_size=10),
            PyannoteSegmenter(strict_limit_duration=30)
        )
    
class _(TranscriberPipeline, register_as='stt-ru-fastconformer-vad'):
    def init(self):
        return LongformVAD(
            NvidiaNemoWrapper(
                'nvidia/stt_ru_fastconformer_hybrid_large_pc',
                beam_size=16,
            ),
            PyannoteSegmenter(strict_limit_duration=30)
        )
    
class _(TranscriberPipeline, register_as='stt-ru-conformer-ctc-large-vad'):
    def init(self):
        return LongformVAD(
            NvidiaNemoWrapper('nvidia/stt_ru_conformer_ctc_large'),
            PyannoteSegmenter(strict_limit_duration=30)
        )

# Quasi-streaming

_name = 'stt-ru-conformer-ctc-large-vad-quasi-streaming'
class _(TranscriberPipeline, register_as=_name):
    def init(self):
        return OfflineToStreaming(
            NvidiaNemoWrapper('nvidia/stt_ru_conformer_ctc_large'), interval=1
        )