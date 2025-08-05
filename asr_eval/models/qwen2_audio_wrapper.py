from typing import cast, override

import torch
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

from ..utils.types import FLOATS
from .base.interfaces import ContextualTranscriber


class Qwen2AudioWrapper(ContextualTranscriber):
    '''
    Qwen2-Audio transcriber.
    
    If domain_text is specified, it is added into prompt with a note "may be related".
    
    Authors: Muharyam Baviev & Oleg Sedukhin
    '''
    def __init__(self, domain_text: str = ''):
        self.processor = Qwen2AudioProcessor.from_pretrained( # type: ignore
            'Qwen/Qwen2-Audio-7B',
            trust_remote_code=True,
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained( # type: ignore
            'Qwen/Qwen2-Audio-7B',
            device_map='cuda',
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).eval()
        self.domain_text = domain_text

    @override
    def contextual_transcribe(
        self, waveform: FLOATS, prev_transcription: str = ''
    ) -> str:
        prompt = (
            'Verbatim and accurately transcribe the following audio recording into Russian'
            ' without adding comments, without correcting the speaker\'s mistakes, without'
            ' translating the text into other languages, without adding new words, and without'
            ' losing words. Your answer must contain only the final transcription text in'
            ' Russian, without losing words or adding new words.'
        )
        if self.domain_text:
            prompt += f' Words from the following text may appear in text: "{self.domain_text}".'
        if prev_transcription:
            prompt += f' The previous transcription was: "{prev_transcription}".'
        conversation = [{
            'role': 'user',
            'content': [{'type': 'text', 'text': prompt}, {'type': 'audio'}],
        }]
        prompt = self.processor.apply_chat_template( # type: ignore
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor( # type: ignore
            text=[prompt], audios=[waveform], return_tensors='pt', padding=True
        ).to(self.model.device)

        generate_ids = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
        )

        # TODO can this be done automatically?
        input_token_len = inputs.input_ids.shape[1] # type: ignore
        generate_ids = generate_ids[:, input_token_len:] # type: ignore
        
        text = cast(str, self.processor.batch_decode( # type: ignore
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]).strip()
        
        return text