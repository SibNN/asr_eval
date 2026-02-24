# Class TimedTranscriber (defined in asr_eval/models/base/interfaces.py at lines 45-66)

class TimedTranscriber(asr_eval.models.base.interfaces.Transcriber):
    """An abstract timed transcriber (audio -> timed text chunks) to
    evaluate on any dataset.

    Overrides a
    :meth:`~asr_eval.models.base.interfaces.Transcriber.transcribe`
    method by concatenating the test chunks by space. Subclasses may
    custoimize this.
    """
    ...

    @abc.abstractmethod
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
        """Transcribes a float32 waveform, typically normalized
        from -1 to 1, into a list of texts with timings. Typically
        the texts are to be concatenated via space, so leading or
        trailing spaces in each chunk are not required.
        """
        ...

    @typing.override
    def transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> str:
    ...