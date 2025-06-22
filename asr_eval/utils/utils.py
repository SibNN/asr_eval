from __future__ import annotations

from datetime import timedelta
import uuid
from typing import Any, TypeVar
from dataclasses import dataclass, asdict, field
from itertools import groupby
from collections.abc import Iterable

import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches
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


def groupby_into_spans(iterable: Iterable[T]) -> Iterable[tuple[T, int, int]]:
    '''
    Find spans of the same value in a sequence. Returns (value, start_index, end_index).
    
    list(groupby_enumerate(['x', 'x', 'b', 'a', 'a', 'a']))
    >>> [('x', 0, 2), ('b', 2, 3), ('a', 3, 6)]
    '''
    for key, group in groupby(enumerate(iterable), key=lambda x: x[1]):
        group = list(group)
        yield key, group[0][0], group[-1][0] + 1


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
    for fmt, start, end in groupby_into_spans(fmt_per_char):
        result += colored(text[start:end], **asdict(fmt))
    
    return result


def draw_line_with_ticks(
    x1: float,
    x2: float,
    y: float,
    y_tick_width: float,
    ax: plt.Axes,
    **kwargs: Any,
):
    ax.plot([x1, x2], [y, y], **kwargs,) # type: ignore
    ax.plot([x1, x1], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore
    ax.plot([x2, x2], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore


def draw_bezier(
    xy_points: list[tuple[float, float]],
    ax: plt.Axes,
    indent: float = 0.1,
    zorder: int = 0,
    lw: float = 1,
    color: str = 'darkgray',
):
    verts: list[tuple[float, float]] = []
    for x, y in xy_points:
        for d in (-indent, 0, indent):
            verts.append((x + d, y))
    verts = verts[1:-1]
    codes = [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.CURVE4] * (len(verts) - 1)
    path = matplotlib.path.Path(verts, codes) # type: ignore
    patch = matplotlib.patches.PathPatch(
        path,
        facecolor='none',
        lw=lw,
        edgecolor=color,
        zorder=zorder
    )
    ax.add_patch(patch)
    # ax.autoscale()


# def get_or_create_ax(figsize: tuple[float, float] = (6, 6)) -> plt.Axes:
#     if not plt.get_fignums():
#         plt.figure(figsize=figsize) # type: ignore
#     return plt.gca()


# def draw_horizontal_interval(
#     x1: float,
#     x2: float,
#     y: float,
#     color: str = 'C0',
#     lw: float = 0.2,
#     ax: plt.Axes | None = None
# ):
#     ax = ax or plt.gca()
#     ax.add_patch(matplotlib.patches.Rectangle(
#         (x1, y - lw / 2),
#         (x2 - x1),
#         lw,
#         linewidth=1,
#         color=color,
#     ))
#     # extend_lims(xmin=x1, xmax=x2, ymin=y - lw / 2, ymax=y + lw / 2, ax=ax)

# def draw_circle(
#     x: float,
#     y: float,
#     color: str = 'C0',
#     s: float = 10,
#     ax: plt.Axes | None = None
# ):
#     ax = ax or plt.gca()
#     plt.scatter([x], [y], s=s, c=color) # type: ignore
#     # extend_lims(xmin=x, xmax=x, ymin=y, ymax=y, ax=ax)