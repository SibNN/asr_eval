from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

import numpy as np

class DataclassDataFrame[T: DataclassInstance]:
    """A pandas-like table backed by a list of rows as dataclass
    objects. That is, :code:`DataclassDataFrame(lst: list[MyDataclass])`
    behaves similarly to
    :code:`pd.DataFrame([vars(obj) for obj in lst])`.
    
    There is no "index" in the DataclassDataFrame, just as in Polars.
    """
    
    def __init__(self, data: list[T] | None = None):
        self.data = data or []
    
    def filter(
        self, **kwargs: Any
    ) -> DataclassDataFrame[T]:
        return DataclassDataFrame[T]([
            row for row in self.data
            if all(getattr(row, key) == val for key, val in kwargs.items())
        ])
    
    def groupby(
        self, by: str | Sequence[str]
    ) -> list[tuple[Any, DataclassDataFrame[T]]]:
        by = [by] if isinstance(by, str) else list(by)
        groups: dict[Any, list[T]] = defaultdict(list)
        for row in self.data:
            key = tuple(getattr(row, f) for f in by)
            if len(key) == 1:
                key = key[0]
            groups[key].append(row)
        return [
            (key, DataclassDataFrame(rows)) for key, rows in groups.items()
        ]
    
    def sort_values(
        self, by: str, ascending: bool = True,
    ) -> DataclassDataFrame[T]:
        order = np.argsort(self[by])
        if not ascending:
            order = order[::-1]
        return DataclassDataFrame[T]([self.data[i] for i in order])
    
    def __getitem__(self, key: str) -> tuple[Any, ...]:
        """Column access. The returned value is immutable."""
        
        return tuple(getattr(row, key) for row in self.data)
    
    def __len__(self) -> int:
        return len(self.data)