# Class BaseStorage (defined in asr_eval/utils/storage/base_storage.py at lines 18-107)

class BaseStorage(abc.ABC):
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
    ...

    @abc.abstractmethod
    def has_row(self, **keys: asr_eval.utils.storage.base_storage.VALUE) -> bool:
        """Checks if we have a row (key-value pair) with the specified
        keys, and omitted keys being "not set".
        """
        ...

    @abc.abstractmethod
    def add_row(self, value: typing.Any, overwrite: bool = True, **keys: asr_eval.utils.storage.base_storage.VALUE):
        """Adds a row (key-value pair) with the specified keys, and
        omitted keys being "not set". If such a row exists, i. e.
        :code:`contains(**keys)` is True, will overwrite if
        :code:`overwrite=True`, otherwise raises :code:`ValueError`.
        """
        ...

    @abc.abstractmethod
    def get_row(self, **keys: asr_eval.utils.storage.base_storage.VALUE) -> typing.Any:
        """Gets a row (key-value pair) with the specified keys, and
        omitted keys being "not set". If such a row does not exist, i.
        e. :code:`contains(**keys)` is False, raises :code:`KeyError`.
        """
        ...

    @abc.abstractmethod
    def delete_row(self, missing_ok: bool = False, **keys: asr_eval.utils.storage.base_storage.VALUE):
        """Removes a row (key-value pair) with the specified keys, and
        omitted keys being "not set". If missing_ok is False and such a
        row does not exist, i. e. :code:`contains(**keys)` is False,
        raises :code:`KeyError`.
        """
        ...

    @abc.abstractmethod
    def list_all(
        self,
        load_values: bool = False,
        **keys: asr_eval.utils.storage.base_storage.VALUE,
    ) -> pl.DataFrame:
        """Gets a list of rows (key-value pairs) with the specified
        keys, and any values for the omitted keys. Fills the "not set"
        values with None. Drops full-None columns.
        """
        ...

    @abc.abstractmethod
    def iter_rows(
        self,
        load_values: bool = False,
        **keys: asr_eval.utils.storage.base_storage.VALUE,
    ) -> collections.abc.Iterator[dict[str, typing.Any]]:
        """Same as :code:`.list_all()`, but returns rows one by one,
        instead of converting all the rows in a dataframe.
        """
        ...

    @abc.abstractmethod
    def delete_all(self, **keys: asr_eval.utils.storage.base_storage.VALUE):
        """Removes all rows (key-value pair) with the specified keys,
        and any values for the omitted keys.
        """
        ...

    @abc.abstractmethod
    def close(self):
        """Closes the storage."""
        ...