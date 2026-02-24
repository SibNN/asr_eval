# Class StreamingToOffline (defined in asr_eval/streaming/wrappers.py at lines 24-67)

class StreamingToOffline(asr_eval.models.base.interfaces.Transcriber):
    """A wrapper that turns
    :class:`~asr_eval.streaming.model.StreamingASR` into a
    :class:`~asr_eval.models.base.interfaces.Transcriber`. Transcribes
    the full audio and returns the full transcription.

    The :code:`StreamingASR` keeps running after :code:`.transcribe`
    and waits for new input streams. You may want to stop it via
    :code:`.streaming_model.stop_thread()` at the end.
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...