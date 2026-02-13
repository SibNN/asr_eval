from __future__ import annotations
from pathlib import Path

from asr_eval.bench.pipelines._registry import TranscriberPipeline
from asr_eval.bench.pipelines._registered.gigaam import VOSK_LM
from asr_eval.ctc.lm import CTCDecoderWithLM
from asr_eval.models.base.longform import LongformCTC
from asr_eval.models.wav2vec2_wrapper import Wav2vec2Wrapper


class _(TranscriberPipeline, register_as='wav2vec2-large-xlsr-53-russian'):
    def init(self):
        return LongformCTC(
            Wav2vec2Wrapper('jonatasgrosman/wav2vec2-large-xlsr-53-russian')
        )


class _(TranscriberPipeline, register_as='wav2vec2-xls-r-1b-russian'):
    def init(self):
        return LongformCTC(
            Wav2vec2Wrapper('jonatasgrosman/wav2vec2-xls-r-1b-russian')
        )


class _(TranscriberPipeline, register_as='wav2vec2-large-ru-golos'):
    def init(self):
        return LongformCTC(Wav2vec2Wrapper('bond005/wav2vec2-large-ru-golos'))


class _(TranscriberPipeline, register_as='wav2vec2-large-ru-golos-lm-bond005'):
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
            LongformCTC(
                Wav2vec2Wrapper('bond005/wav2vec2-large-ru-golos-with-lm')
            ),
            kenlm_path=kenlm_path,
            unigrams=Path(unigrams_path).read_text().splitlines(),
            alpha=0.4612,
            beta=0.3271,
        )


class _(TranscriberPipeline, register_as='wav2vec2-large-ru-golos-lm-t-one'):
    def init(self):
        from huggingface_hub import hf_hub_download # type: ignore
        return CTCDecoderWithLM(
            LongformCTC(
                Wav2vec2Wrapper('bond005/wav2vec2-large-ru-golos-with-lm')
            ),
            hf_hub_download('t-tech/T-one', 'kenlm.bin'),
        )


_name = 'wav2vec2-large-ru-golos-lm-vosk-0.42'
class _(TranscriberPipeline, register_as=_name):
    def init(self):
        return CTCDecoderWithLM(
            LongformCTC(
                Wav2vec2Wrapper('bond005/wav2vec2-large-ru-golos-with-lm')
            ),
            VOSK_LM,
        )