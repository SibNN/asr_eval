from __future__ import annotations

from dataclasses import asdict, dataclass, field

from termcolor import colored

from .misc import groupby_into_spans


@dataclass
class Formatting:
    color: str | None = None
    on_color: str | None = None
    attrs: set[str] = field(default_factory=set)

    def _update_inplace(self, other: Formatting):
        if other.color is not None:
            self.color = other.color
        if other.on_color is not None:
            self.on_color = other.on_color
        for attr in other.attrs:
            self.attrs.add(attr)


@dataclass
class FormattingSpan:
    fmt: Formatting
    start: int
    end: int


def apply_ansi_formatting(text: str, spans: list[FormattingSpan]) -> str:
    '''
    Applies ANSI formatting to the specified spans in the text. Example:

    ```
    apply_ansi_formatting('ABCDEFXXXYYY', [
        FormattingSpan(Formatting(color='red'), 0, 5),
        FormattingSpan(Formatting(on_color='on_black'), 0, 3),
        FormattingSpan(Formatting(attrs={'strike'}), 0, 9),
    ])
    ```

    Results in: '\x1b[9m\x1b[31mABCDE\x1b[0m\x1b[9mFXXX\x1b[0mYYY\x1b[0m'
    (can be rendered in jupyter notebook or console)

    If overlaps happen, shorter spans are prioritized.
    '''
    fmt_per_char: list[Formatting] = [Formatting() for _ in range(len(text))]
    for span in sorted(spans, key=lambda span: span.end - span.start)[::-1]:
        for i in range(span.start, span.end):
            fmt_per_char[i]._update_inplace(span.fmt)  # pyright:ignore[reportPrivateUsage]

    result = ''
    for fmt, start, end in groupby_into_spans(fmt_per_char):
        result += colored(text[start:end], **asdict(fmt))

    return result