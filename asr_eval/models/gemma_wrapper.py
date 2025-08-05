from typing import Any, cast, override

import torch
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForConditionalGeneration
from transformers.models.gemma3n.processing_gemma3n import Gemma3nProcessor

from ..utils.types import FLOATS
from .base.interfaces import Transcriber


class Gemma3nWrapper(Transcriber):
    def __init__(self):
        self.processor = cast(
            Gemma3nProcessor,
            Gemma3nProcessor.from_pretrained('google/gemma-3n-E4B-it') # type: ignore
        )
        self.model = Gemma3nForConditionalGeneration.from_pretrained( # type: ignore
            'google/gemma-3n-E4B-it',
            device_map='cuda',
            torch_dtype=torch.float16,
        ).eval()

    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # processor calls self.feature_extractor(audio, ...), it trims audio to 30 seconds
        assert len(waveform) <= 16_000 * 30, 'Audio should be <= 30 seconds length'
        
        prompt = (
            'Транскрибируй только русскую речь из аудио. Проанализируй ВСЁ аудио, если в'
            ' начале нет речи - не думай, что её там вообще нет. Если нет речи - верни'
            ' пустую строку. Не комментируй.'
        )
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": waveform},
                {"type": "text", "text": prompt},
            ]
        }]
        
        # if we remove tokenize=True, we can see the resulting prompt, it will be like this:
        # '<bos><start_of_turn>user\n<audio_soft_token>{prompt}<end_of_turn>\n<start_of_turn>model\n'
        inputs = cast(dict[str, Any], self.processor.apply_chat_template(  
            conversation,
            add_generation_prompt=True,  # this adds <start_of_turn>model\n
            tokenize=True,
            return_tensors='pt',
            return_dict=True,
        ).to(self.model.device, dtype=self.model.dtype)) # type: ignore
        
        generate_ids = self.model.generate(   # type: ignore
            **inputs,
            max_new_tokens=1024,
            # do_sample=False,
            # temperature=0.0,
            # TODO select hyperparams
            do_sample=True,
            num_beams=2,
            temperature=0.7,
            repetition_penalty=1.2,
        )
        
        # TODO can this be done automatically?
        input_token_len = inputs.input_ids.shape[1] # type: ignore
        generate_ids = generate_ids[:, input_token_len:] # type: ignore
        
        text = cast(str, self.processor.batch_decode(   # type: ignore
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]).strip()
        return text