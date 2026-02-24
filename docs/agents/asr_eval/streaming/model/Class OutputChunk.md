# Class OutputChunk (defined in asr_eval/streaming/model.py at lines 86-115)

@dataclasses.dataclass(kw_only=True)
class OutputChunk:
    """ An output chunk for
    :class:`~asr_eval.streaming.model.StreamingASR`. Output chunks are
    sent by :code:`StreamingASR` thread and received manually or
    by :func:`~asr_eval.streaming.evaluation.receive_transcription`.

    See :class:`~asr_eval.streaming.model.StreamingASR` and
    :func:`~asr_eval.streaming.evaluation.receive_transcription` docs
    for usage details.
    """
    ...

    data: asr_eval.streaming.model.TranscriptionChunk | typing.Literal[Signal.FINISH]
    """Either a part of transcription, or a :code:`Signal.FINISH`."""

    seconds_processed: float
    """A total audio seconds processed before emitting the current
    chunk. Is filled by the transcriber.
    """

    put_timestamp: float = np.nan
    """Is filled automatically when the chunk is added to the output
    buffer.
    """

    get_timestamp: float = np.nan
    """Is filled automatically when the chunk is taken from the output
    buffer.
    """