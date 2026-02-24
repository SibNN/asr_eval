# Function get_spans_for_char_aligned (defined in asr_eval/align/char_aligner.py at lines 118-154)

def get_spans_for_char_aligned(
    al: asr_eval.align.char_aligner.CharAligned
) -> list[tuple[int, int]]:
    """
    Locates positions in the
    :class:`~asr_eval.align.char_aligner.CharAligned` where both texts
    contain space. Splits by these positions, and returns the resulting
    spans.

    This can be userful to match predictions from two models, where
    there may be cases when a single word from one first model matches
    with two words from another model. In this case the current function
    will return an interval spanning both words.
    """
    ...