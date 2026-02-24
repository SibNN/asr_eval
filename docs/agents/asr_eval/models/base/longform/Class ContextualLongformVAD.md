# Class ContextualLongformVAD (defined in asr_eval/models/base/longform.py at lines 185-240)

class ContextualLongformVAD(asr_eval.models.base.interfaces.TimedTranscriber):
    """A wrapper that is similar to
    :class:`~asr_eval.models.base.longform.LongformVAD`, but for each
    chunk passes the previously transcribed text, up to the
    :code:`max_history_words`, as a context for the next chunk when
    transcribing it.

    Requies a shortform model to be a
    :class:`~asr_eval.models.base.interfaces.ContextualTranscriber`.
    """
    ...

    @typing.override
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
    ...