# Function merge_synthetic_speech (defined in asr_eval/utils/audio_ops.py at lines 156-177)

def merge_synthetic_speech(
    waveforms: list[asr_eval.utils.types.FLOATS],
    sampling_rate: int = 16_000,
    pause_range: tuple[float, float] = (0.2, 1.2),
    random_seed: int | None = None,
) -> asr_eval.utils.types.FLOATS:
    """Merges speech segments using silent pauses of random length in
    :code:`pause_range`.

    Is suitable to construct a longform synthetic speech.
    """
    ...