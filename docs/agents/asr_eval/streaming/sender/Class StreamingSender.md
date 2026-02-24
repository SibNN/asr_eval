# Class StreamingSender (defined in asr_eval/streaming/sender.py at lines 117-274)

@dataclasses.dataclass
class StreamingSender:
    """ Can be used to automate sending audio stream to
    :class:`~asr_eval.streaming.model.StreamingASR`.

    Args:
        cutoffs: A schedule to send the audio.
        waveform: A waveform in float32 dtype. The sampling rate
            information is encoded in cutoffs, because they store both
            the audio time and the audio array position.
        asr: A streaming transcriber to send chunks into.
        id: A recording ID to assign.
        verbose: Whether to print each chunk info to stdout.

    Call
    :meth:`~asr_eval.streaming.sender.StreamingSender.start_sending()`
    to start a sending process.

    Note:
        Keeps the history of all sent chunks for evaluation purposes
        (can be retrieved with
        :meth:`~asr_eval.streaming.sender.StreamingSender.join_and_get_history`).
        To avoid out of memory, ensure that senders are
        garbage-collected afterwards.
    """
    ...

    cutoffs: list[asr_eval.streaming.sender.Cutoff]

    waveform: asr_eval.utils.types.FLOATS

    asr: asr_eval.streaming.model.StreamingASR

    id: asr_eval.streaming.buffer.ID_TYPE = field(default_factory=new_uid)

    verbose: bool = False

    sampling_rate: int = 16_000

    @property
    def audio_length_sec(self) -> float:
    ...

    def start_sending(self, without_delays: bool = False) -> typing.Self:
        """If :code:`without_delays=False` (default) starts sending in a
        separate thread according to the shedule given in constuctor.
        If :code:`without_delays=True` sends all the chunks immediately.
        Non-blocking.
        """
        ...

    def join(self):
        """Wait for the sending process to finish."""
        ...

    def join_and_get_history(self) -> list[asr_eval.streaming.model.InputChunk]:
        """Wait for the sending process to finish and return the history
        of chunks sent.
        """
        ...

    def get_status(self) -> typing.Literal['not_started', 'started', 'finished']:
        """ Possible statuses:

        - not_started: Sending was not started.
        - started: Sending in progress.
        - finished: Sending finished.
        """
        ...