from __future__ import annotations

import uuid
from typing import TypeVar
from itertools import groupby
from collections.abc import Iterable


__all__ = [
    'new_uid',
    'groupby_into_spans',
    'self_product_nonequal',
]


def new_uid() -> str:
    '''
    A unique ids generator
    '''
    return str(uuid.uuid4())


T = TypeVar('T')


def groupby_into_spans(iterable: Iterable[T]) -> Iterable[tuple[T, int, int]]:
    '''
    Find spans of the same value in a sequence. Returns (value, start_index, end_index).
    
    list(groupby_enumerate(['x', 'x', 'b', 'a', 'a', 'a']))
    >>> [('x', 0, 2), ('b', 2, 3), ('a', 3, 6)]
    '''
    for key, group in groupby(enumerate(iterable), key=lambda x: x[1]):
        group = list(group)
        yield key, group[0][0], group[-1][0] + 1


# def check_not_none(val: T | None) -> T:
#     """
#     Checks that the variable is not None and returs it, useful to calm down type checker.
#     """
#     # https://discuss.python.org/t/improve-typing-with-to-force-not-none-value/7840/15
#     assert val is not None
#     return val


def self_product_nonequal(iterable: Iterable[T], triangle: bool) -> Iterable[tuple[T, T]]:
    '''
    Given an iterable X, yields all elements from cartesian product X * X, excluding
    the main diagonal. If `triangle=True`, also excludes the upper triangle.
    '''
    for x1_idx, x1 in enumerate(iterable):
        for x2_idx, x2 in enumerate(iterable):
            if (
                (triangle and x1_idx < x2_idx)
                or (not triangle and x1_idx != x2_idx)
            ):
                yield (x1, x2)