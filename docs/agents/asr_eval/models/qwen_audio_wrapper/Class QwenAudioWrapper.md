# Class QwenAudioWrapper (defined in asr_eval/models/qwen_audio_wrapper.py at lines 19-80)

class QwenAudioWrapper(asr_eval.models.base.interfaces.Transcriber):
    """A wrapper for Qwen-Audio v1 (NOTE: not v2!). Experimental, may
    not work.

    Requires :code:`transformers` package.
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...