# Class WhisperLongformWrapper (defined in asr_eval/models/whisper_wrapper.py at lines 12-89)

class WhisperLongformWrapper(asr_eval.models.base.interfaces.Transcriber):
    """A wrapper for Whisper.

    If audio is long, internally performs a longform transcription and
    passes the previously transcriber words each time.

    Since the transcription history is used internally in
    :code:`WhisperForConditionalGeneration.generate`, this class does
    not implement a
    :class:`~asr_eval.models.base.interfaces.ContextualTranscriber`
    interface.
    """
    ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
        # https://github.com/huggingface/transformers/pull/27658
        ...