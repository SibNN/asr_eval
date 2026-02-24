# Function waveform_as_file (defined in asr_eval/utils/audio_ops.py at lines 178-193)

@contextlib.contextmanager
def waveform_as_file(waveform: asr_eval.utils.types.FLOATS) -> typing.Iterator[pathlib.Path]:
    """Turns a waveform into a file. The file is deleted on exit from
    the context.

    Example:
        >>> with waveform_as_file(waveform) as audio_path:  # doctest: +SKIP
        ...     recognize_speech(path=audio_path)
    """
    ...