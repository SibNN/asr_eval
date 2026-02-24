# Class FormattingSpan (defined in asr_eval/utils/formatting.py at lines 41-52)

@dataclasses.dataclass
class FormattingSpan:
    """A Formatting with the corresponding start and end positions in
    the text.

    Note that the positions are specified for the text before adding
    ANSI color codes.
    """
    ...

    fmt: asr_eval.utils.formatting.Formatting

    start: int

    end: int