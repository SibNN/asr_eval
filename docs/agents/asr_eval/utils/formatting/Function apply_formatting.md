# Function apply_formatting (defined in asr_eval/utils/formatting.py at lines 54-91)

def apply_formatting(
    text: str,
    spans: list[asr_eval.utils.formatting.FormattingSpan],
    color_mode: typing.Literal['ansi', 'html'] = 'ansi',
) -> str:
    """Applies ANSI formatting to the specified spans in the text.

    Example:
        >>> from asr_eval.utils.formatting import apply_formatting, Formatting, FormattingSpan
        >>> apply_formatting('ABCDEFXXXYYY', [ # doctest: +SKIP
        ...     FormattingSpan(Formatting(color='red'), 0, 5),
        ...     FormattingSpan(Formatting(on_color='on_black'), 0, 3),
        ...     FormattingSpan(Formatting(attrs={'strike'}), 0, 9),
        ... ])
        \x1b[9m\x1b[40m\x1b[31mABC\x1b[0m\x1b[9m\x1b
        [31mDE\x1b[0m\x1b[9mFXXX\x1b[0mYYY\x1b[0m

    (this can be rendered in Jupyter notebook or console)

    If :code:`color_mode='html'`, converts the ANSI codes into HTML.
    If overlaps occur, the shorter spans are prioritized.
    """
    ...