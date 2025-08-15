from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, overload

import numpy as np
import pandas as pd


T = TypeVar('T')
T2 = TypeVar('T2')

class Table2D(Generic[T]):
    '''
    A type-safe 2D table with cells of type T with default cell values.
    
    Supports:
    - slicing [:, :] - returns a new Table2D[T]
    - slicing [:, i] or [i, :] - returns a list[T]
    - slicing [i, j] - returns T
    - mapping with function T -> T2, returns a new Table2D[T2]
    - appending and prepending rows and columns of type list[T]
    - converting to DataFrame (without col/row names) with .to_pandas()
    - getting .shape
    '''
    
    _data: np.ndarray[tuple[int, int], Any]
    
    def __init__(self, data: np.ndarray[tuple[int, int], Any]):
        self._data = data
    
    def to_numpy(self) -> np.ndarray[tuple[int, int], Any]:
        return self._data.copy()
    
    @classmethod
    def construct(cls, rows: int, cols: int, default: Callable[[], T]):
        # cannot use np.array([...]) because it will result in shape (X, y, 0) if default=list
        _data = np.empty((rows, cols), dtype=object) 
        for i in range(rows):
            for j in range(cols):
                _data[i, j] = default()
        return cls(_data)
    
    @overload
    def __getitem__(self, idx: tuple[int, int]) -> T: ...
    
    @overload
    def __getitem__(self, idx: tuple[slice, slice]) -> Table2D[T]: ...
    
    @overload
    def __getitem__(self, idx: tuple[int, slice]) -> list[T]: ...
    
    @overload
    def __getitem__(self, idx: tuple[slice, int]) -> list[T]: ...

    def __getitem__(self, idx: tuple[int | slice, int | slice]) -> T | Table2D[T] | list[T]:
        idx1, idx2 = idx
        is_slice1 = isinstance(idx1, slice)
        is_slice2 = isinstance(idx2, slice)
        if is_slice1 and is_slice2:
            # table
            return self.__class__(self._data[idx1, idx2]) # type: ignore
        elif not is_slice1 and not is_slice2:
            # cell value
            return self._data[idx1, idx2]
        else:
            # list
            return self._data[idx1, idx2].tolist()
    
    def __setitem__(self, idx: tuple[int, int], value: T):
        self._data[idx] = value
    
    def prepend_row(self, row: list[T]):
        self._data = np.concatenate([self._list_to_np(row)[None, :], self._data], axis=0) # type: ignore
    
    def append_row(self, row: list[T]):
        self._data = np.concatenate([self._data, self._list_to_np(row)[None, :]], axis=0) # type: ignore
    
    def prepend_col(self, col: list[T]):
        self._data = np.concatenate([self._list_to_np(col)[:, None], self._data], axis=1) # type: ignore
    
    def append_col(self, col: list[T]):
        self._data = np.concatenate([self._data, self._list_to_np(col)[:, None]], axis=1) # type: ignore
        
    def _list_to_np(self, lst: list[T]) -> np.ndarray[tuple[Any], Any]:
        # cannot use np.array(lst) because it will try to unpack if T is a list
        result = np.empty(len(lst), dtype=object)
        for i, val in enumerate(lst):
            result[i] = val
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape # type: ignore

    def __repr__(self) -> str:
        shape = f'{self._data.shape[0]}x{self._data.shape[1]}'
        return f"Table[{shape}]<{T.__name__ if hasattr(T, '__name__') else T}>"

    def map(self, fn: Callable[[T], T2]) -> Table2D[T2]:
        rows, cols = self._data.shape
        result = np.empty((rows, cols), dtype=object) 
        for i in range(rows):
            for j in range(cols):
                result[i, j] = fn(self._data[i, j])
        return Table2D[T2](result)

    def map_with_indices(self, fn: Callable[[int, int, T], T2]) -> Table2D[T2]:
        rows, cols = self._data.shape
        result = np.empty((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = fn(i, j, self._data[i, j])
        return Table2D[T2](result)
    
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)