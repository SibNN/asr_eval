from typing import Any, Literal, cast, override

import torch

from ..utils.types import FLOATS
from .base.interfaces import ContextualTranscriber


__all__ = [
    'Gemma3nWrapper',
]


class Gemma3nWrapper(ContextualTranscriber):
    '''
    Gemma3n transcriber.
    
    If domain_text is specified, it is added into prompt with a note "may be related".
    
    Authors: Timur Rafikov & Oleg Sedukhin
    '''
    def __init__(self, lang: Literal['en', 'ru'] = 'ru', domain_text: str = ''):
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nForConditionalGeneration
        from transformers.models.gemma3n.processing_gemma3n import Gemma3nProcessor
        self.processor = cast(
            Gemma3nProcessor,
            Gemma3nProcessor.from_pretrained('google/gemma-3n-E4B-it') # type: ignore
        )
        self.model = Gemma3nForConditionalGeneration.from_pretrained( # type: ignore
            'google/gemma-3n-E4B-it',
            device_map='cuda',
            torch_dtype=torch.float16,
        ).eval()
        self.domain_text = domain_text
        self.lang: Literal['en', 'ru'] = lang

    @override
    def contextual_transcribe(
        self, waveform: FLOATS, prev_transcription: str = ''
    ) -> str:
        # processor calls self.feature_extractor(audio, ...), it trims audio to 30 seconds
        assert len(waveform) <= 16_000 * 30, 'Audio should be <= 30 seconds length'
        
        match self.lang:
            case 'en':
                prompt = (
                    'Transcribe only English speech from audio. Analyze ALL audio, if there is no'
                    ' speech at the beginning - don\'t assume that there is none at all. If there'
                    ' is no speech - return an empty string. Do not add any comments.'
                )
                if self.domain_text:
                    prompt += f' The speech may contain words from the following text: "{self.domain_text}".'
                if prev_transcription:
                    prompt += f' The transcription of the previous part was as follows: "{prev_transcription}".'
            case 'ru':
                prompt = (
                    'Транскрибируй только русскую речь из аудио. Проанализируй ВСЁ аудио, если в'
                    ' начале нет речи - не думай, что её там вообще нет. Если нет речи - верни'
                    ' пустую строку. Не комментируй.'
                )
                if self.domain_text:
                    prompt += f' В речи могут встретиться слова из следующего текста: "{self.domain_text}".'
                if prev_transcription:
                    prompt += f' Транскрипция предыдущей части была следующей: "{prev_transcription}".'
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