# Class CannotFillTimings (defined in asr_eval/align/timings.py at lines 93-100)

class CannotFillTimings(ValueError):
    """ An exception raised from
    :func:`~asr_eval.align.timings.fill_word_timings_inplace` that
    indicates a failure to fill timings, usually because of absence of
    the required characters in the model vocab.
    """
    ...