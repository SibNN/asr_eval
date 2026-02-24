# Function transcribe_parallel (defined in asr_eval/streaming/caller.py at lines 32-70)

def transcribe_parallel(
    asr: asr_eval.streaming.model.StreamingASR,
    waveforms: list[asr_eval.utils.types.FLOATS],
    n_threads: int,
    send_all_without_delays: bool = False,
    real_time_interval_sec: float = 1 / 25,
    speed_multiplier: float = 1,
) -> dict[asr_eval.streaming.buffer.ID_TYPE, list[asr_eval.streaming.model.OutputChunk]]:
    """Transcribes the waveforms in parallel, but with no more than
    :code:`n_threads` simultaneous senders.

    Call :code:`asr.start_thread()` before calling this method, and
    :code:`asr.stop_thread()` after.
    """
    ...