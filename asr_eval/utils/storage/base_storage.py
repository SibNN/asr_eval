from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

from tqdm.auto import tqdm


VALUE = str | int | float | bool
"""A value for key-value pairs that act as a joint key in
:class:`~asr_eval.utils.storage.BaseStorage`.
"""


class BaseStorage(ABC):
    """A persistent key-value storage.
    
    Represents a table, where rows are key-value pairs, the "value"
    column stores any picklable objects, and a variable number of
    columns act as a joint key, with values of type string, int, float,
    bool or None (not set).
    
    To add a new row (key-value pair), you don't need to specify values
    for all the key columns added earlier, the omitted columns will be
    filled with None. If you add a new key-value pair with a new key
    column not present earlier, we add this column with a value None for
    all other rows.
    
    Note that since we do not differentiate bewteen the explicit "null"
    and the "not set", storing the explicit nulls is not possible.
    
    Example:
        >>> from asr_eval.utils.storage import BaseStorage, ShelfStorage
        >>> st: BaseStorage = ShelfStorage('tmp/storage.db')
        >>> st.add_row(value='Hi', dataset='fleurs', sample=0, what='ground_truth')
        >>> st.add_row(value='Hi', dataset='fleurs', model='whisper', sample=0, what='pred')
        >>> st.add_row(value='Ho', dataset='fleurs', model='tuned', steps=100, sample=0, what='pred')
        >>> storage.list_all(load_values=True)  # doctest: +SKIP
    
    The result will be a dataframe with 3 rows and columns 'value',
    'dataset', 'sample', 'model', 'type', 'steps'. Cell values for the
    omitted keys will be filled with None.
    """
    
    @abstractmethod
    def has_row(self, **keys: VALUE) -> bool:
        """Checks if we have a row (key-value pair) with the specified
        keys, and omitted keys being "not set".
        """
    
    @abstractmethod
    def add_row(self, value: Any, overwrite: bool = True, **keys: VALUE):
        """Adds a row (key-value pair) with the specified keys, and
        omitted keys being "not set". If such a row exists, i. e.
        :code:`contains(**keys)` is True, will overwrite if
        :code:`overwrite=True`, otherwise raises :code:`ValueError`.
        """
    
    @abstractmethod
    def get_row(self, **keys: VALUE) -> Any:
        """Gets a row (key-value pair) with the specified keys, and
        omitted keys being "not set". If such a row does not exist, i.
        e. :code:`contains(**keys)` is False, raises :code:`KeyError`.
        """
    
    @abstractmethod
    def delete_row(self, missing_ok: bool = False, **keys: VALUE):
        """Removes a row (key-value pair) with the specified keys, and
        omitted keys being "not set". If missing_ok is False and such a
        row does not exist, i. e. :code:`contains(**keys)` is False,
        raises :code:`KeyError`.
        """
    
    @abstractmethod
    def list_all(
        self,
        load_values: bool = False,
        **keys: VALUE,
    ) -> pl.DataFrame:
        """Gets a list of rows (key-value pairs) with the specified
        keys, and any values for the omitted keys. Fills the "not set"
        values with None. Drops full-None columns.
        """
    
    @abstractmethod
    def iter_rows(
        self,
        load_values: bool = False,
        **keys: VALUE,
    ) -> Iterator[dict[str, Any]]:
        """Same as :code:`.list_all()`, but returns rows one by one,
        instead of converting all the rows in a dataframe.
        """
    
    @abstractmethod
    def delete_all(self, **keys: VALUE):
        """Removes all rows (key-value pair) with the specified keys,
        and any values for the omitted keys.
        """
    
    @abstractmethod
    def close(self):
        """Closes the storage."""


def map_column_values(
    storage: BaseStorage,
    column_name: str,
    value_mapping: list[tuple[VALUE, VALUE]],
):
    print('Listing all rows...')
    rows = storage.list_all()
    n_replaced = 0
    for keys in tqdm(rows.iter_rows(named=True), total=len(rows)):
        if column_name in keys:
            current_value = keys[column_name]
            for from_value, to_value in value_mapping:
                if (
                    current_value == from_value
                    and type(current_value) == type(from_value)
                ):
                    new_keys = keys | {column_name: to_value}
                    value = storage.get_row(**keys)
                    storage.delete_row(missing_ok=False, **keys)
                    storage.add_row(value=value, overwrite=False, **new_keys)
                    n_replaced += 1
                    break
    print(f'Replaced {n_replaced} rows')