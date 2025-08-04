from typing import Any, Literal, cast, override

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from ..utils.audio_ops import waveform_as_file
from ..utils.types import FLOATS
from .base.interfaces import Transcriber


QWEN_AUDIO_LANGUAGES = Literal['en', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'ko']

class QwenAudioWrapper(Transcriber):
    def __init__(self, lang: QWEN_AUDIO_LANGUAGES = 'en', audio_lang_unknown: bool = False):
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
        
        audio_lang = 'unknown' if audio_lang_unknown else lang
        self.instruct_tokens = (
            f'<|startoftranscription|><|{audio_lang}|><|transcribe|>'
            f'<|{lang}|><|notimestamps|><|wo_itn|>'
        )
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        with waveform_as_file(waveform) as audio_path:
            query = f'<audio>{audio_path}</audio>{self.instruct_tokens}'
            audio_info = cast(dict[str, Any] | None, self.tokenizer.process_audio(query)) # type: ignore
            
        assert audio_info is not None

        inputs = cast(
            dict[str, Any],
            self.tokenizer(query, return_tensors='pt', audio_info=audio_info).to(self.model.device) # type: ignore
        )

        pred = cast(torch.Tensor, self.model.generate(**inputs, audio_info=audio_info)) # type: ignore
        text = cast(str, self.tokenizer.decode( # type: ignore
            pred.cpu()[0], # type: ignore
            skip_special_tokens=True,
            audio_info=audio_info,
        ))
        return text.split('<|startoftranscription|>')[-1]  # skip_special_tokens=True does not remove it