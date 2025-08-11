from typing import Callable, Generic, TypeVar

import numpy as np


T = TypeVar('T')

class Table2D(Generic[T]):
    '''
    A type-safe 2D table with cells of type T with default cell values.
    '''
    
    def __init__(self, rows: int, cols: int, default: Callable[[], T]):
        self._data = np.empty((rows, cols), dtype=object)
        # cannot instantiate as np.array([ ... ]) due to the resulting
        # shape (8, 79, 0) if default=list
        for i in range(rows):
            for j in range(cols):
                self._data[i, j] = default()

    def __getitem__(self, idx: tuple[int, int]) -> T:
        return self._data[idx]

    def __setitem__(self, idx: tuple[int, int], value: T) -> None:
        self._data[idx] = value

    def shape(self) -> tuple[int, int]:
        return self._data.shape # type: ignore

    def __repr__(self) -> str:
        shape = f'{self._data.shape[0]}x{self._data.shape[1]}'
        return f"Table[{shape}]<{T.__name__ if hasattr(T, '__name__') else T}>"