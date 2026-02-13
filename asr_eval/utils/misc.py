from __future__ import annotations
import uuid
from itertools import groupby, chain
from collections.abc import Iterable

import numpy as np

from asr_eval.utils.types import FLOATS, INTS


__all__ = [
    'new_uid',
    'groupby_into_spans',
    'list_join',
    'rolling_window',
    'locate_subarray_in_array',
]


def new_uid() -> str:
    """A unique ID generator."""
    return str(uuid.uuid4())[:16]


def groupby_into_spans[T](
    iterable: Iterable[T]
) -> Iterable[tuple[T, int, int]]:
    """Find spans of the same value in a sequence. Returns (value,
    start_index, end_index).
    
    Example:
        >>> list(groupby_into_spans(['x', 'x', 'b', 'a', 'a', 'a']))
        [('x', 0, 2), ('b', 2, 3), ('a', 3, 6)]
    """
    
    for key, group in groupby(enumerate(iterable), key=lambda x: x[1]):
        group = list(group)
        yield key, group[0][0], group[-1][0] + 1


def list_join[T](sep: T, iterable: Iterable[T]) -> list[T]:
    """Combines iterables via a given separator. Acts like str.join,
    but for lists.
    """
    
    it = iter(iterable)
    try:
        first = [next(it)]
    except StopIteration:
        return []
    return list(chain(
        first,
        *([sep, x] for x in it),
    ))


def rolling_window[T: (INTS, FLOATS)](arr: T, size: int) -> T:
    """Returns all subarrays of length :code:`size`, stacked together
    along a new axis.
    
    Example:
        >>> rolling_window(np.array([1, 0, 2, 1, 3, 5]), 3) # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 2], [0, 2, 1], [2, 1, 3], [1, 3, 5]])
       
    Taken from: https://stackoverflow.com/a/7100681
    """
    
    shape = arr.shape[:-1] + (arr.shape[-1] - size + 1, size)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides) # type: ignore


def locate_subarray_in_array[T: (INTS, FLOATS)](
    arr: T, subarr: T
) -> list[int]:
    """Finds all positions X where :code:`arr[X:X+len(subarr)]` equals
    :code:`subarr`, in effiecient way.
    """
    
    assert arr.ndim == 1
    assert subarr.ndim == 1
    mask = np.all(rolling_window(arr, size=len(subarr)) == subarr, axis=-1)
    indices, = np.where(mask)
    return indices.tolist()