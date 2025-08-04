from typing import Literal, override
from openai import OpenAI
from pydantic_extra_types.language_code import LanguageAlpha2

from .base.openai_wrapper import APITranscriber


class VoxtralWrapper(APITranscriber):
    '''
    Calls Voxtral via OpenAI API.
    
    Example:
    
    voxtral = VoxtralWrapper('mistralai/Voxtral-Mini-3B-2507')
    text = voxtral.transcribe(speech_sample(repeats=2))
    print(text)
    voxtral.stop_vllm_server()
    
    See the VLLM source code in vllm.model_executor.models.voxtral
    
    According to VoxtralEncoderModel.prepare_inputs_for_conv, the Voxtral pipeline splits a long audio
    into non-overlapping chunks, then processes each chunk via Whisper and concatenate the outputs.
    So, the LLM sees the whole long audio at once.
    '''
    def __init__(
        self,
        model_name: str = 'mistralai/Voxtral-Mini-3B-2507',
        client: OpenAI | Literal['run_local_server'] = 'run_local_server',
        language: str | LanguageAlpha2 = 'ru',
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        super().__init__(
            client=client,
            model_name=model_name,
            language=language,
            temperature=temperature
        )
    
    @override
    def vllm_run_args(self) -> list[str]:
        return [
            'vllm',
            'serve',
            self.model_name,
            '--tokenizer_mode',
            'mistral',
            '--config_format',
            'mistral',
            '--load_format',
            'mistral',
            '--tensor-parallel-size',
            '1',
            '--tool-call-parser',
            'mistral',
            '--enable-auto-tool-choice',
            '--gpu-memory-utilization',
            '0.75',
        ]