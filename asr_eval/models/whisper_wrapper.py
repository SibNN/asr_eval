from typing import Literal, override, cast

from asr_eval.models.base.interfaces import Transcriber
from asr_eval.utils.types import FLOATS


__all__ = [
    'WhisperLongformWrapper',
]


class WhisperLongformWrapper(Transcriber):
    """A wrapper for Whisper.
    
    If audio is long, internally performs a longform transcription and
    passes the previously transcriber words each time.
    
    Since the transcription history is used internally in
    :code:`WhisperForConditionalGeneration.generate`, this class does
    not implement a
    :class:`~asr_eval.models.base.interfaces.ContextualTranscriber`
    interface.
    """
    
    def __init__(
        self,
        model_name: str = 'openai/whisper-large-v3',
        preproc_name: str | None = None,
        lang: Literal['russian', 'english'] | None = None,
        condition_on_prev_tokens: bool = False,
        temperature: float = 0,
        dtype: str = 'float32',
    ):
        import torch
        from transformers import (
            WhisperForConditionalGeneration, WhisperProcessor
        )
        
        self.model_name = model_name
        self.lang = lang
        self.condition_on_prev_tokens = condition_on_prev_tokens
        self.temperature = temperature
        
        self.whisper_processor = cast(
            WhisperProcessor,
            WhisperProcessor.from_pretrained( # type: ignore
                preproc_name or self.model_name,
                # how this is used? maybe 'ru', 'en'?
                language=self.lang,
                task='transcribe',
            )
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = WhisperForConditionalGeneration.from_pretrained( # type: ignore
            self.model_name,
            attn_implementation='sdpa',
            torch_dtype=dtype,
        ).to(self.device) # type: ignore
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # https://github.com/huggingface/transformers/pull/27658
        inputs = self.whisper_processor( # type: ignore
            waveform,
            return_tensors='pt',
            truncation=False,
            padding='max_length',
            # TODO probably we do not need this for Whisper
            return_attention_mask=True,
            sampling_rate=16_000
        ).to(self.model.dtype) # type: ignore
        result = self.model.generate( # type: ignore
            **inputs.to(self.model.device), # type: ignore
            condition_on_prev_tokens=self.condition_on_prev_tokens,
            # temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            temperature=self.temperature,
            do_sample=self.temperature > 0,  # ?? OK?
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            return_timestamps=True,  # required for longform
            language=f'<|{self.lang}|>' if self.lang else None,
            task='transcribe',
            forced_decoder_ids=None,
        )
        return self.whisper_processor.batch_decode( # type: ignore
            result, skip_special_tokens=True # type: ignore
        )[0]