from __future__ import annotations
from pathlib import Path

from asr_eval import CACHE_DIR
from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.ctc.lm import CTCDecoderWithLM
from asr_eval.models.base.longform import LongformCTC, LongformVAD
from asr_eval.models.pyannote_vad import PyannoteSegmenter
from asr_eval.models.gigaam_wrapper import (
    GigaAMShortformCTC, GigaAMShortformRNNT
)
from asr_eval.streaming.wrappers import OfflineToStreaming


# e2e

class _(TranscriberPipeline, register_as='gigaam3-ctc-e2e'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC('v3_e2e'))

class _(TranscriberPipeline, register_as='gigaam3-rnnt-e2e-vad'):
    def init(self):
        return LongformVAD(GigaAMShortformRNNT('v3_e2e'), PyannoteSegmenter())

# CTC uniform

class _(TranscriberPipeline, register_as='gigaam2-ctc'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC('v2'))

class _(TranscriberPipeline, register_as='gigaam2-ctc-20'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC('v2'), segment_shift=20)

class _(TranscriberPipeline, register_as='gigaam2-ctc-blank'):
    def init(self):
        return LongformCTC(
            GigaAMShortformCTC('v2'),
            averaging_weights='blank_sep',
            segment_shift=20,
        )

class _(TranscriberPipeline, register_as='gigaam3-ctc-blank'):
    def init(self):
        return LongformCTC(
            GigaAMShortformCTC('v3'),
            averaging_weights='blank_sep',
            segment_shift=20,
        )

class _(TranscriberPipeline, register_as='gigaam2-ctc-fp16'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC('v2', fp16=True))

class _(TranscriberPipeline, register_as='gigaam3-ctc'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC('v3'))

class _(TranscriberPipeline, register_as='gigaam3-ctc-fp16'):
    def init(self):
        return LongformCTC(GigaAMShortformCTC('v3', fp16=True))

# CTC with VAD

class _(TranscriberPipeline, register_as='gigaam2-ctc-vad'):
    def init(self):
        return LongformVAD(GigaAMShortformCTC('v2'), PyannoteSegmenter())

# RNNT with VAD

class _(TranscriberPipeline, register_as='gigaam2-rnnt-vad'):
    def init(self):
        return LongformVAD(GigaAMShortformRNNT('v2'), PyannoteSegmenter())

class _(TranscriberPipeline, register_as='gigaam3-rnnt-vad'):
    def init(self):
        return LongformVAD(GigaAMShortformRNNT('v3'), PyannoteSegmenter())

# CTC uniform + LM

VOSK_LM = str(CACHE_DIR / 'vosk-model-ru-0.42-compile/db/ru.lm.gz')



class _(TranscriberPipeline, register_as='gigaam2-ctc-lm-vosk-0.42'):
    # to install Vosk LM, run installation/vosk_lm.sh
    def init(self):
        return CTCDecoderWithLM(
            LongformCTC(GigaAMShortformCTC('v2')),
            VOSK_LM,
        )

class _(TranscriberPipeline, register_as='gigaam2-ctc-lm-t-one'):
    def init(self):
        from huggingface_hub import hf_hub_download # type: ignore
        return CTCDecoderWithLM(
            LongformCTC(GigaAMShortformCTC('v2')),
            hf_hub_download('t-tech/T-one', 'kenlm.bin'),
        )

class _(TranscriberPipeline, register_as='gigaam3-ctc-lm-t-one'):
    def init(self):
        from huggingface_hub import hf_hub_download # type: ignore
        return CTCDecoderWithLM(
            LongformCTC(GigaAMShortformCTC(version='v3')),
            hf_hub_download('t-tech/T-one', 'kenlm.bin'),
        )

class _(TranscriberPipeline, register_as='gigaam3-ctc-blank-lm-t-one'):
    def init(self):
        from huggingface_hub import hf_hub_download # type: ignore
        return CTCDecoderWithLM(
            LongformCTC(
                GigaAMShortformCTC('v3'),
                averaging_weights='blank_sep',
                segment_shift=20,
            ),
            hf_hub_download('t-tech/T-one', 'kenlm.bin'),
        )

class _(TranscriberPipeline, register_as='gigaam2-ctc-lm-bond005'):
    def init(self):
        from huggingface_hub import hf_hub_download # type: ignore
        kenlm_path = hf_hub_download(
            'bond005/wav2vec2-large-ru-golos-with-lm',
            'language_model/ru_3gram.bin'
        )
        unigrams_path = hf_hub_download(
            'bond005/wav2vec2-large-ru-golos-with-lm',
            'language_model/unigrams.txt'
        )
        return CTCDecoderWithLM(
            LongformCTC(GigaAMShortformCTC('v2')),
            kenlm_path=kenlm_path,
            unigrams=Path(unigrams_path).read_text().splitlines(),
            alpha=0.4612,
            beta=0.3271,
        )

# CTC with VAD + LM

class _(TranscriberPipeline, register_as='gigaam2-ctc-vad-lm-vosk-0.42'):
    # to install Vosk LM, run installation/vosk_lm.sh
    def init(self):
        return LongformVAD(
            CTCDecoderWithLM(
                GigaAMShortformCTC('v2'),
                VOSK_LM,
            ),
            PyannoteSegmenter(),
        )

# Quasi-streaming

class _(TranscriberPipeline, register_as='gigaam2-ctc-quasi-streaming'):
    def init(self):
        return OfflineToStreaming(GigaAMShortformCTC('v2'), interval=1)