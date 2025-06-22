from __future__ import annotations

from datetime import timedelta
import uuid
from typing import TypeVar
from itertools import groupby
from collections.abc import Iterable

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


