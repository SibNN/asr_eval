from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, cast, override

if TYPE_CHECKING:
    import torch

from asr_eval.utils.audio_ops import waveform_as_file
from asr_eval.utils.types import FLOATS
from asr_eval.models.base.interfaces import Transcriber


__all__ = [
    'QwenAudioWrapper',
]


QWEN_AUDIO_LANGUAGES = Literal['en', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'ko']

class QwenAudioWrapper(Transcriber):
    """A wrapper for Qwen-Audio v1 (NOTE: not v2!). Experimental, may
    not work.
    
    Requires :code:`transformers` package.
    """
    
    def __init__(
        self,
        language: QWEN_AUDIO_LANGUAGES = 'en',
        audio_lang_unknown: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained( # type: ignore
            'Qwen/Qwen-Audio',
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained( # type: ignore
            'Qwen/Qwen-Audio',
            device_map='cuda',
            trust_remote_code=True,
            bf16=True,
        ).eval()
        
        audio_lang = 'unknown' if audio_lang_unknown else language
        self.instruct_tokens = (
            f'<|startoftranscription|><|{audio_lang}|><|transcribe|>'
            f'<|{language}|><|notimestamps|><|wo_itn|>'
        )
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        with waveform_as_file(waveform) as audio_path:
            # requires soundfile lib
            query = f'<audio>{audio_path}</audio>{self.instruct_tokens}'
            audio_info = cast(
                dict[str, Any] | None,
                self.tokenizer.process_audio(query) # type: ignore
            )
            
        assert audio_info is not None

        inputs = cast(
            dict[str, Any],
            self.tokenizer( # type: ignore
                query, return_tensors='pt', audio_info=audio_info
            ).to(self.model.device) # type: ignore
        )

        pred: torch.Tensor = self.model.generate( # type: ignore
            **inputs, audio_info=audio_info
        )
        text = cast(str, self.tokenizer.decode( # type: ignore
            pred.cpu()[0], # type: ignore
            skip_special_tokens=True,
            audio_info=audio_info,
        ))
        
        # the setting `skip_special_tokens=True` does not remove this
        return text.split('<|startoftranscription|>')[-1]