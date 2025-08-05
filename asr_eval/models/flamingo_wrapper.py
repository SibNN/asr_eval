import sys
from typing import Literal, override

from huggingface_hub import snapshot_download # type: ignore

from ..utils.audio_ops import waveform_as_file
from ..utils.types import FLOATS
from .base.interfaces import Transcriber


class FlamingoWrapper(Transcriber):
    '''
    Flamingo transcriber.
    
    Authors: Dmitry Ezhov & Oleg Sedukhin
    '''
    def __init__(self, lang: Literal['en', 'ru'] = 'ru'):
        code_path = snapshot_download(repo_id='nvidia/audio-flamingo-3', repo_type='space')
        
        sys.path.append(code_path)
        import llava # type: ignore
        self.llava_module = llava
        sys.path.pop(-1)
        
        model_path = snapshot_download(repo_id='nvidia/audio-flamingo-3')
        self.model = llava.load(model_path, model_base=None).cuda() # type: ignore
        self.lang: Literal['en', 'ru'] = lang

    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # processor calls self.feature_extractor(audio, ...), it trims audio to 30 seconds
        assert len(waveform) <= 16_000 * 30, 'Audio should be <= 30 seconds length'
        
        with waveform_as_file(waveform) as audio_path:
            # keep the file until generation is done
            sound = self.llava_module.Sound(str(audio_path))
            match self.lang:
                case 'en':
                    prompt = 'Transcribe the audio.'
                case 'ru':
                    prompt = 'Транскрибируй аудио на русском языке. Текст должен быть на русском языке.'
            return self.model.generate_content( # type: ignore
                [sound, f'<sound>\n{prompt}'],
                generation_config=self.model.default_generation_config, # type: ignore
            )