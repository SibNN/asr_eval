# Class InputChunk (defined in asr_eval/streaming/model.py at lines 56-84)

@dataclasses.dataclass(kw_only=True)
class InputChunk:
    """ An input chunk for
    :class:`~asr_eval.streaming.model.StreamingASR`. Input chunks can be
    sent by :class:`~asr_eval.streaming.sender.StreamingSender` or
    manually and received by :code:`StreamingASR` thread.

    See :class:`~asr_eval.streaming.model.StreamingASR` docs for usage
    details.
    """
    ...

    data: asr_eval.streaming.model.AUDIO_CHUNK_TYPE | typing.Literal[Signal.FINISH]
    """Either a chunk of audio stream, or a :code:`Signal.FINISH`."""

    end_time: float
    """A chunk end time (in seconds) in the audio timescale, where 0
    means the beginning of the audio recording.
    """

    put_timestamp: float = np.nan
    """Is filled automatically when the chunk is added to the
    input buffer.
    """

    get_timestamp: float = np.nan
    """Is filled automatically when the :code:`StreamingASR` thread
    takes the chunk from the buffer.
    """