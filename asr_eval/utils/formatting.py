from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Literal

from termcolor import colored
from ansi2html import Ansi2HTMLConverter
from ansi2html.style import get_styles

from asr_eval.utils.misc import groupby_into_spans


__all__ = [
    'Formatting',
    'FormattingSpan',
    'apply_formatting',
]


@dataclass
class Formatting:
    """ ANSI text formatting attrubutes, such as "bold", "red" etc.
    
    Example:
        >>> from asr_eval.utils.formatting import Formatting
        >>> Formatting(color='red', attrs={'strike'}) # doctest: +ELLIPSIS
        ...
    """
    color: str | None = None
    on_color: str | None = None
    attrs: set[str] = field(default_factory=set[str])

    def _update_inplace(self, other: Formatting):
        if other.color is not None:
            self.color = other.color
        if other.on_color is not None:
            self.on_color = other.on_color
        for attr in other.attrs:
            self.attrs.add(attr)


@dataclass
class FormattingSpan:
    """A Formatting with the corresponding start and end positions in
    the text.
    
    Note that the positions are specified for the text before adding
    ANSI color codes.
    """
    fmt: Formatting
    start: int
    end: int


def apply_formatting(
    text: str,
    spans: list[FormattingSpan],
    color_mode: Literal['ansi', 'html'] = 'ansi',
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
    
    fmt_per_char: list[Formatting] = [Formatting() for _ in range(len(text))]
    for span in sorted(spans, key=lambda span: span.end - span.start)[::-1]:
        for i in range(span.start, span.end):
            fmt_per_char[i]._update_inplace(span.fmt)  # pyright:ignore[reportPrivateUsage]

    ansi_colored_text = ''
    for fmt, start, end in groupby_into_spans(fmt_per_char):
        ansi_colored_text += colored(text[start:end], **asdict(fmt))

    match color_mode:
        case 'ansi':
            return ansi_colored_text
        case 'html':
            return ansi_to_html(ansi_colored_text)


_ansi_converter: Ansi2HTMLConverter | None = None

def ansi_to_html(ansi_text: str) -> str:
    """Converts ANSI color codes to HTML.
    
    Is based on ansi2html library. Returns first a :code:`<style>` tag
    with the required styles, then the
    :code:`<span style="white-space: pre">` tag with the colored
    elements from `ansi_text`.
    """
    
    global _ansi_converter
    _ansi_converter = _ansi_converter or Ansi2HTMLConverter(dark_bg=True)
    attrs = _ansi_converter.prepare(ansi_text, ensure_trailing_newline=False)
    all_styles = get_styles(
        _ansi_converter.dark_bg,
        _ansi_converter.line_wrap,
        _ansi_converter.scheme,
    )
    backgrounds = all_styles[:5]
    used_styles = [
        s for s in all_styles if s.klass.lstrip('.') in attrs['styles']
    ]
    styles = ' '.join([str(x) for x in backgrounds + used_styles])
    return (
        f'<style>{styles}</style>'
        f'<span style="white-space: pre">{attrs["body"]}</span>'
    )