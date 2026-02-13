from __future__ import annotations
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from transformers.models.whisper.feature_extraction_whisper import (
        WhisperFeatureExtractor
    )
    from transformers.generation.utils import GenerationMixin

from asr_eval.models.base.interfaces import Transcriber
from asr_eval.utils.types import FLOATS

__all__ = [
    'VikhrBorealisWrapper',
]


class VikhrBorealisWrapper(Transcriber):
    """A Vikhr Borealis wrapper.
    
    Loading a model takes a long time, around 2 min.
    
    Installation: see :doc:`/guide_installation` page.
    """
    
    model: GenerationMixin
    # tokenizer: Qwen2TokenizerFast
    extractor: WhisperFeatureExtractor
    
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoFeatureExtractor

        model_name = 'Vikhrmodels/Borealis'
        self.model = AutoModelForCausalLM.from_pretrained( # type: ignore
            model_name, trust_remote_code=True
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name) # type: ignore

        self.model.eval() # type: ignore
        self.model = self.model.to('cuda') # type: ignore

    @override
    def transcribe(self, waveform: FLOATS) -> str:
        import torch
        
        proc = self.extractor(
            waveform,
            sampling_rate=16_000,
            padding='max_length',
            max_length=16_000 * 30,
            return_attention_mask=True,
            return_tensors='pt',
        )
        with torch.inference_mode():
            return self.model.generate( # type: ignore
                mel=proc.input_features.squeeze(0).to('cuda'), # type: ignore
                att_mask=proc.attention_mask.squeeze(0).to('cuda'), # type: ignore
                max_new_tokens=350,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.2,
            )