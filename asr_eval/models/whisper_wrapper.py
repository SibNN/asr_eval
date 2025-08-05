from typing import Literal, override, cast

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base.interfaces import Transcriber
from ..utils.types import FLOATS


class WhisperLongformWrapper(Transcriber):
    def __init__(
        self,
        model_name: str = 'openai/whisper-large-v3',
        lang: Literal['ru', 'en'] = 'ru',
        condition_on_prev_tokens: bool = False,
    ):
        self.model_name = model_name
        self.lang = lang
        self.condition_on_prev_tokens = condition_on_prev_tokens
        
        self.whisper_processor = cast(WhisperProcessor, WhisperProcessor.from_pretrained( # type: ignore
            self.model_name,
            language='Russian' if self.lang == 'ru' else 'English', # how this is used? maybe 'ru', 'en'?
            task='transcribe',
        ))
        
        self.model = WhisperForConditionalGeneration.from_pretrained( # type: ignore
            self.model_name,
            attn_implementation='sdpa',
            torch_dtype=torch.float32
        ).cuda() # type: ignore
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # https://github.com/huggingface/transformers/pull/27658
        inputs = self.whisper_processor( # type: ignore
            waveform,
            return_tensors='pt',
            truncation=False,
            padding='longest',
            return_attention_mask=True,  # probably we do not need this for Whisper
            sampling_rate=16_000
        )
        result = self.model.generate( # type: ignore
            **inputs.to(self.model.device), # type: ignore
            condition_on_prev_tokens=self.condition_on_prev_tokens,
            # temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            temperature=0, # for determinism
            do_sample=False,  # for determinism
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            return_timestamps=True,  # required foir longform
            language=f'<|{self.lang}|>',
            task='transcribe',
            forced_decoder_ids=None,
        )
        return self.whisper_processor.batch_decode( # type: ignore
            result, skip_special_tokens=True # type: ignore
        )[0]