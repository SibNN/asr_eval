# Class Cutoff (defined in asr_eval/streaming/sender.py at lines 23-60)

@dataclasses.dataclass(slots=True)
class Cutoff:
    """A container for audio position and a real time (relative) moment.
    Is used to shedule a waveform sending into
    :class:`~asr_eval.streaming.model.StreamingASR`.

    Let we have an audio `waveform` and two consecutive cutoffs
    :code:`[c1, c2]`:

    .. code-block:: python

        [Cutoff(tr1, ta1, pos1), Cutoff(tr2, ta2, pos2)]

    This means that:

    1. :code:`waveform[:c1.arr_pos]` gives an audio with length
       :code:`c1.t_audio`.
    2. :code:`waveform[:c2.arr_pos]` gives an audio with length
       :code:`c2.t_audio`.
    1. :code:`waveform[c1.arr_pos:c2.arr_pos]` should be sent at the
       time :code:`c2.t_real`.
    """
    ...

    t_real: float
    """A real world time measured from the beginning of the sending
    process.
    """

    t_audio: float
    """A time in the audio timescale. For example, if we send 10 sec
    audio in 5 seconds (with 2x speed), the :code:`t_real` will be 5 in
    the end, and :code:`t_audio` will be 10.
    """

    arr_pos: int
    """A position in the audio as array. Is calculated from the
    :code:`t_audio` using array length per second.
    """