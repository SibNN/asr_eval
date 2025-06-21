from __future__ import annotations

from datetime import timedelta
import uuid
from typing import TypeVar
from dataclasses import dataclass, asdict, field
from itertools import groupby

from termcolor import colored
import srt


T = TypeVar('T')

def N(val: T | None) -> T:
    """
    Checks that the variable is not None and returs it, useful to calm down type checker.
    """
    # https://discuss.python.org/t/improve-typing-with-to-force-not-none-value/7840/15
    assert val is not None
    return val


def new_uid() -> str:
    return str(uuid.uuid4())


def utterances_to_srt(utterances: list[tuple[str, float, float]]) -> str:
    '''
    Composes an SRT file.
    '''
    return srt.compose([ # type: ignore
        srt.Subtitle(
            index=None,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text
        )
        for text, start, end in utterances
    ], reindex=True)


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
    for fmt, group in groupby(enumerate(fmt_per_char), key=lambda x: x[1]):
        group = list(group)
        start, end = group[0][0], group[-1][0]
        block = colored(text[start:end + 1], **asdict(fmt))
        result += block
    
    return result