# Function convert_audio_format (defined in asr_eval/utils/audio_ops.py at lines 127-154)

def convert_audio_format(
    waveform: asr_eval.utils.types.FLOATS,
    to_audio_type: typing.Literal['float', 'int', 'bytes', 'wav'] = 'float',
) -> asr_eval.utils.types.FLOATS | asr_eval.utils.types.INTS | bytes:
    """
    Converts a waveform with sampling rate 16000 into one of the
    pre-defined formats:

    - 'float': float values, preferrably from -1 to 1. Does nothing
        because this is the same as input format.
    - 'int': :code:`np.int16` values.
    - 'bytes': 2 bytes per frame.
    - 'wav': 2 bytes per frame plus WAV header.

    TODO find some python library that already supports these formats
    and conversions, or design this better.
    """
    ...