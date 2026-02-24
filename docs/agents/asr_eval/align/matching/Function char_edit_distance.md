# Function char_edit_distance (defined in asr_eval/align/matching.py at lines 26-32)

@functools.cache
def char_edit_distance(true: str, pred: str) -> int:
    """A :code:`@cache` wrapper for `nltk.edit_distance`. Calculates
    character edit distance between strings.
    """
    ...