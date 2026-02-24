# Function make_sender (defined in asr_eval/streaming/evaluation.py at lines 101-161)

def make_sender(
    waveform: asr_eval.utils.types.FLOATS,
    asr: asr_eval.streaming.model.StreamingASR,
    real_time_interval_sec: float = 1 / 5,
    speed_multiplier: float = 1,
    uid: str | None = None,
) -> tuple[list[asr_eval.streaming.sender.Cutoff], asr_eval.streaming.sender.StreamingSender]:
    """An automation to make a sender that sends an audio recording into
    a :class:`~asr_eval.streaming.model.StreamingASR`.

    After running :code:`cutoffs, sender = make_sender(...)`, you
    typically need to run :code:`sender.start_sending()` to start a
    thread that actually sends all the chunks.

    Args:
        waveform: The audio in float32 dtype with sampling rate 16000.
            Note that the streaming recognizer may accept a different
            sampling rate or dtype. A conversion to the required rate
            and dtype will be done on-the-fly inside this function.
        asr: A streaming transcriber to send chunks into.
        real_time_interval_sec: How often in real time to send chunks?
        speed_multiplier: For example, if :code:`speed_multiplier=2`,
            will sent the audio twice of normal speed, that is, a 10
            seconds audio will be sent in 5 seconds.
        uid: Assign UID to the recording (select ranfom of omitted).

    Returns:
        - A sending schedule in form of a list of cutoffs. See
            :func:`~asr_eval.streaming.sender.get_uniform_cutoffs` for
            details.
        - A sender object thatis ready to start sending. Call
        :meth:`~asr_eval.streaming.sender.StreamingSender.start_sending`
        to start sending chunks.
    """
    ...