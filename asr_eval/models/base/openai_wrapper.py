from typing import Any, Literal, cast, override
import io

from openai import OpenAI, NotGiven, NOT_GIVEN
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionTokenLogprob
from openai.types.audio import Transcription
from openai.types.audio.transcription_create_params import ChunkingStrategyVadConfig
from openai.types.audio.transcription import Logprob
from mistral_common.protocol.instruct.messages import UserContentChunk, TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio
from pydantic_extra_types.language_code import LanguageAlpha2

from .interfaces import Transcriber
from ...utils.server import ServerAsSubprocess
from ...utils.audio_ops import waveform_to_bytes
from ...utils.types import FLOATS


class APITranscriber(Transcriber):
    '''
    A connector to OpenAI API for audio LLMs. Runs via `client.audio.transcriptions.create`.
    This class wraps api_transcribe() to implement Transcriber interface.
    See api_transcribe() docstring for chunking_strategy and temperature params.
    
    This class also allows to auto-start a local VLLM server. To do this, subclass this class
    and define vllm_run_args(). See VoxtralWrapper as the example.
    
    Example with starting VLLM manually:
    
    1. Start a local VLLM server

    vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral \\
        --load_format mistral --tensor-parallel-size 1 --tool-call-parser mistral \\
        --enable-auto-tool-choice --gpu-memory-utilization 0.75

    2. Run the code

    from openai import OpenAI
    from asr_eval.utils.audio_ops import speech_sample
    from asr_eval.models.base.openai_wrapper import do_transcribe

    transcriber = APITranscriber(
        OpenAI(api_key='EMPTY', base_url='http://localhost:8000/v1'),
        model_name='mistralai/Voxtral-Mini-3B-2507',
        language='ru',
    )
    
    waveform = speech_sample(repeats=5)
    transcriber.transcribe(waveform)
    '''
    def __init__(
        self,
        model_name: str = 'mistralai/Voxtral-Mini-3B-2507',
        client: OpenAI | Literal['run_local_server'] = 'run_local_server',
        language: str | LanguageAlpha2 = 'ru',
        prompt: str | None = None,
        chunking_strategy: Literal['auto'] | ChunkingStrategyVadConfig | NotGiven = NOT_GIVEN,
        temperature: float = 0.7,
        local_server_verbose: bool = False,
    ):
        if client == 'run_local_server':
            self.vllm_proc = ServerAsSubprocess(
                self.vllm_run_args() + ['--host', '127.0.0.1', '--port', '8001'],
                ready_message='Application startup complete',
                verbose=local_server_verbose,
            )
            client = OpenAI(api_key='EMPTY', base_url='http://localhost:8001/v1')
        else:
            self.vllm_proc = None
        
        self.model_name = model_name
        self.client = client
        self.language = language
        self.prompt = prompt
        self.chunking_strategy: Literal['auto'] | ChunkingStrategyVadConfig | NotGiven = chunking_strategy
        self.temperature = temperature
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        text, _ = api_transcribe(
            client=self.client,
            waveform=waveform,
            language=self.language,
            model_name=self.model_name,
            prompt=self.prompt,
            chunking_strategy=self.chunking_strategy,
            temperature=self.temperature,
            
        )
        return text
    
    def vllm_run_args(self) -> list[str]:
        return [
            'vllm',
            'serve',
            self.model_name,
        ]
    
    def stop_vllm_server(self):
        assert self.vllm_proc
        self.vllm_proc.stop()

  
def api_transcribe(
    client: OpenAI,
    waveform: FLOATS,
    model_name: str,
    language: str | LanguageAlpha2 = 'en',
    prompt: str | None = None,
    chunking_strategy: Literal['auto'] | ChunkingStrategyVadConfig | NotGiven = NOT_GIVEN,
    temperature: float = 0.7,
) -> tuple[str, list[Logprob] | None]:
    '''
    A connector to OpenAI API for audio LLMs. Runs via `client.audio.transcriptions.create`.
    See the full usage example in APITranscriber (wrapper around api_transcribe).
    
    Sends a message with audio and language to transcribe. A default temperature is 0.7,
    this value is taken from mistral_common BaseCompletionRequest
    
    Returns:
    - a transcription
    - logprobs if returned by the model
    
    According to openai.types.audio.transcription.Transcription docstring, logprobs
    are returned only with the models `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`
    
    By default chunking_strategy is unset, and the audio is transcribed as a single block,
    according to `client.audio.transcriptions.create` docstring.
    
    Voxtral seem to ignore both chunking_strategy and a request to return logprobs, according
    to VLLM server logs.
    
    Raises:
    - openai.APIConnectionError if cannot connect to the API
    - openai.NotFoundError if cannot find the specified model_name
    - InternalServerError in some cases (happened with VseGPT)
    '''
    # flac is actually a zipped (lossess) wav, should have smaller size than wav
    file = io.BytesIO(waveform_to_bytes(waveform, sampling_rate=16_000, format='flac'))
    response: Transcription = client.audio.transcriptions.create(
        file=file,
        prompt=prompt if prompt is not None else NOT_GIVEN,
        temperature=temperature,
        model=model_name,
        language=LanguageAlpha2(language),
        chunking_strategy=chunking_strategy,
        include=['logprobs'],
    )
    return response.text, response.logprobs


def api_chat_completion(
    client: OpenAI,
    waveform: FLOATS,
    model_name: str,
    logprobs: bool = False,
    pre_prompt: str | None = None,
    post_prompt: str | None = None,
    **generate_kwargs: Any,
) -> tuple[str, bool, list[ChatCompletionTokenLogprob] | None]:
    '''
    A connector to OpenAI API for audio LLMs. Runs via `client.chat.completions.create`.
    
    Sends a message with audio, optional pre-prompt and post-prompt.
    
    Returns:
    - a model response
    - a flag if the response was truncated bue to the length limits
    - logprobs if logprobs=True
    
    Raises:
    - ContentFilterException if the model refuses to generate
    - openai.APIConnectionError if cannot connect to the API
    - openai.NotFoundError if cannot find the specified model_name
    '''
    # mistral_common.audio.Audio is a container with data, sampling rate, and format
    # flac is actually a zipped (lossess) wav, should have smaller size than wav
    audio = Audio.from_bytes(waveform_to_bytes(waveform, sampling_rate=16_000, format='flac'))
    
    messages: list[UserContentChunk] = [AudioChunk.from_audio(audio)]
    if pre_prompt is not None:
        messages.insert(0, TextChunk(text=pre_prompt))
    if post_prompt is not None:
        messages.append(TextChunk(text=post_prompt))
    
    user_message = cast(
        ChatCompletionUserMessageParam,
        UserMessage(content=messages).to_openai()
    )
    response = client.chat.completions.create(
        messages=[user_message],
        model=model_name,
        stream=False,
        n=1,
        logprobs=logprobs,
        **generate_kwargs,
    )
    
    choice = response.choices[0]
    assert choice.finish_reason in ('stop', 'length', 'content_filter'), choice.finish_reason
    if choice.finish_reason == 'content_filter':
        raise ContentFilterException()
    
    returned_logprobs = None
    if logprobs:
        assert choice.logprobs is not None and choice.logprobs.content is not None
        returned_logprobs = choice.logprobs.content
    
    return (
        choice.message.content or '',
        choice.finish_reason == 'length',
        returned_logprobs,
    )


class ContentFilterException(RuntimeError):
    pass