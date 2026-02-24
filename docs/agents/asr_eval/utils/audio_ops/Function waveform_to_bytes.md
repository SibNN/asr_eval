# Function waveform_to_bytes (defined in asr_eval/utils/audio_ops.py at lines 46-58)

def waveform_to_bytes(
    waveform: asr_eval.utils.types.FLOATS, sampling_rate: int = 16_000, format: str = 'wav'
) -> bytes:
    """Converts a waveform into WAV bytes (or another format passed as
    :code:`format` argument).
    """
    ...