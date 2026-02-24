# Class ContextualTranscriber (defined in asr_eval/models/base/interfaces.py at lines 124-142)

class ContextualTranscriber(asr_eval.models.base.interfaces.Transcriber):
    """An abstract transcriber being able to accept previous
    transcription as a context.
    """
    ...

    @abc.abstractmethod
    def contextual_transcribe(
        self, waveform: asr_eval.utils.types.FLOATS, prev_transcription: str = ''
    ) -> str:
        """Transcribes a float32 waveform, typically normalized
        from -1 to 1. The :code:`prev_transcription` represents a
        transcription from all the previous text before the current
        :code:`waveform`.
        """
        ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...