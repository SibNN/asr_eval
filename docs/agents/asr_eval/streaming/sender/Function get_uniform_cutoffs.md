# Function get_uniform_cutoffs (defined in asr_eval/streaming/sender.py at lines 62-95)

def get_uniform_cutoffs(
    waveform: asr_eval.utils.types.FLOATS,
    real_time_interval_sec: float = 1 / 25,
    speed_multiplier: float = 1.0,
    sampling_rate: int = 16_000,
) -> list[asr_eval.streaming.sender.Cutoff]:
    """Returns a uniform shedule to send the audio.

    Args:
        waveform: The audio in float32 dtype.
        real_time_interval_sec: How often in real time to send chunks?
        speed_multiplier: For example, if :code:`speed_multiplier=2`,
            will sent the audio twice of normal speed, that is, a 10
            seconds audio will be sent in 5 seconds.
        sampling_rate: The sampling rate of the :code:`waveform`.
    """
    ...