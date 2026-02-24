# Class APITranscriber (defined in asr_eval/models/base/openai_wrapper.py at lines 30-141)

class APITranscriber(asr_eval.models.base.interfaces.Transcriber):
    """A connector to OpenAI API for audio LLMs. Runs via
    :code:`client.audio.transcriptions.create`. This class wraps
    :func:`~asr_eval.models.base.openai_wrapper.api_transcribe()` to
    implement :class:`~asr_eval.models.base.interfaces.Transcriber`
    interface. See the
    :func:`~asr_eval.models.base.openai_wrapper.api_transcribe()`
    docstring for :code:`chunking_strategy` and :code:`temperature`
    params.

    This class also allows to auto-start a local VLLM server. To do
    this, subclass this class and define :code:`vllm_run_args()`. See
    :class:`~asr_eval.models.voxtral_wrapper.VoxtralWrapper` as the
    example.

    Example with starting VLLM manually:

    1. Start a local VLLM server

    .. code-block:: python

        vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral \\
            --config_format mistral --load_format mistral \\
            --tensor-parallel-size 1 --tool-call-parser mistral  \\
            --enable-auto-tool-choice --gpu-memory-utilization 0.75

    2. Run the code

    .. code-block:: python

        from openai import OpenAI
        from asr_eval.models.base.openai_wrapper import APITranscriber

        transcriber = APITranscriber(
            OpenAI(api_key='EMPTY', base_url='http://localhost:8000/v1'),
            model_name='mistralai/Voxtral-Mini-3B-2507',
            language='ru',
        )

        waveform = <load you audio sample>
        transcriber.transcribe(waveform)
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...

    def vllm_api_server_args(self) -> list[str]:
    ...

    def vllm_run_args(self) -> list[str]:
    ...

    def stop_vllm_server(self):
    ...