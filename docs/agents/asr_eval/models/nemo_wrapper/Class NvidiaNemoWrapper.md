# Class NvidiaNemoWrapper (defined in asr_eval/models/nemo_wrapper.py at lines 22-149)

class NvidiaNemoWrapper(asr_eval.models.base.interfaces.Transcriber):
    """
    A Nvidia NEMO wrapper

    Installation: see :doc:`/guide_installation` page.

    Some of the available models (many more are available):
    1. "nvidia/canary-1b-v2"
        - NOTE: Specify language, example:
          :code:`inference_kwargs={'source_lang': 'ru', 'target_lang': 'ru'}`
        - NOTE: in Nemo :code:`beam_size=1` by default
    2. "nvidia/parakeet-tdt-0.6b-v3"
        - NOTE: Supports torch.float16 or torch.bfloat16 only with
          :code:`amp=True`
        - NOTE: in Nemo :code:`beam_size=2` by default
    3. "nvidia/stt_ru_fastconformer_hybrid_large_pc"
        NOTE: in Nemo :code:`beam_size=2` by default

    Dtypes:
        - for :code:`amp=True`, available dtypes are torch.float16,
          torch.bfloat16
        - for :code:`amp=False`, available dtypes are torch.float16,
          torch.bfloat16, torch.float32

    Notes:

    This wrapper is build using the following docs and examples:
    https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/transcribe_speech.py
    https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr
        /api.html#nemo.collections.asr.parts.mixins.transcription.TranscriptionMixin

    The NEMO wrapper seems not to perform internal VAD; it raises OOM on
    too long audios. From the EncDecMultiTaskModel docstrings:
    "recommended length per file is between 5 and 25 seconds, but it is
    possible to pass a few hours long file if enough GPU memory is
    available".

    The :code:`.transcribe()` method of the NEMO's
    :code:`TranscriptionMixin` allows to pass :code:`timestamps=True`.
    It raises error for Canary, but returns timestamps for Parakeet and
    FastComformer. However, the output timestamps require postprocessing
    that is not implemented currently.

    Some of the models should support CTC interface and/or LM
    interation, but this is not implemented in asr_eval currently.

    To get the full list of available models, run:

    .. code-block:: python

        from nemo.collections.asr.models import ASRModel
        print(ASRModel.list_available_models())
    """
    ...

    model: TranscriptionMixin

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...