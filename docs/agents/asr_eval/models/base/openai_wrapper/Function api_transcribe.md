# Function api_transcribe (defined in asr_eval/models/base/openai_wrapper.py at lines 143-206)

def api_transcribe(
    client: OpenAI,
    waveform: asr_eval.utils.types.FLOATS,
    model_name: str,
    language: str | LanguageAlpha2 | None = None,
    prompt: str | None = None,
    chunking_strategy: (
        typing.Literal['auto', 'omit'] | ChunkingStrategyVadConfig | Omit
    ) = 'omit',
    temperature: float = 0.7,
    format: str = 'flac',
) -> tuple[str, list[Logprob] | None]:
    """A connector to OpenAI API for audio LLMs. Runs via
    :code:`client.audio.transcriptions.create`. See the full usage
    example in
    :class:`~asr_eval.models.base.openai_wrapper.APITranscriber`.

    Sends a message with audio and language to transcribe. A default
    temperature is 0.7, this value is taken from mistral_common's
    :code:`BaseCompletionRequest`.

    Returns:
        A transcription and logprobs (optional, if returned by the
        model). According to
        :code:`openai.types.audio.transcription.Transcription`
        docstring, logprobs are returned only with the models
        `gpt-4o-transcribe` and `gpt-4o-mini-transcribe`.

    By default :code:`chunking_strategy` is unset, and the audio is
    transcribed as a single block, according to
    :code:`client.audio.transcriptions.create` docstring.

    Voxtral seem to ignore both :code:`chunking_strategy` and a request
    to return logprobs, according to VLLM server logs.

    :code:`format` is FLAC by default, this is actually a compressed
    (lossess) wav, should have smaller size than wav.

    Raises:
        openai.APIConnectionError: If cannot connect to the API
        openai.NotFoundError: If cannot find the specified model_name
        InternalServerError: In some cases (happened with VseGPT)
    """
    ...