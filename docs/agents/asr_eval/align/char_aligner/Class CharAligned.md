# Class CharAligned (defined in asr_eval/align/char_aligner.py at lines 15-25)

@dataclasses.dataclass
class CharAligned:
    """
    A char-level alignment for two texts, obtained by
    :func:`~asr_eval.align.char_aligner.char_align` (see its dostring
    for details).
    """
    ...

    first: str

    matching: str

    second: str