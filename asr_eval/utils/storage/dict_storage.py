from collections.abc import Iterator, MutableMapping
from pathlib import Path
import shelve
from typing import Any, cast, override

import polars as pl

from asr_eval.utils.storage.base_storage import BaseStorage, VALUE


class DictStorage(BaseStorage):
    """A dict-based in-memory :code:`BaseStorage` implementation."""

    _dict: MutableMapping[str, Any]
    # since key is a dict in BaseStorage, _dict maps from
    # _encode_key(key) to a tuple (key, value)

    def __init__(self):
        self._dict = {}
    
    @override
    def has_row(self, **keys: VALUE) -> bool:
        return _encode_key(keys) in self._dict
    
    @override
    def add_row(self, value: Any, overwrite: bool = True, **keys: VALUE):
        key = _encode_key(keys)
        if not overwrite and key in self._dict:
            raise ValueError(f'The key {keys} already exists')
        self._dict[key] = (keys, value)
    
    @override
    def get_row(self, **keys: VALUE) -> Any:
        try:
            return self._dict[_encode_key(keys)][1]
        except KeyError:
            raise KeyError(str(keys)) from None
    
    @override
    def delete_row(self, missing_ok: bool = False, **keys: VALUE):
        key = _encode_key(keys)
        if missing_ok:
            self._dict.pop(key, None)
        else:
            try:
                del self._dict[key]
            except KeyError:
                raise KeyError(str(keys)) from None
    
    @override
    def list_all(
        self,
        load_values: bool = False,
        **keys: VALUE,
    ) -> pl.DataFrame:
        df_rows = list(self.iter_rows(load_values=load_values, **keys))
        return pl.DataFrame(df_rows, schema_overrides={'value': pl.Object()})
    
    @override
    def iter_rows(
        self,
        load_values: bool = False,
        **keys: VALUE,
    ) -> Iterator[dict[str, Any]]:
        for row_keys_as_str, row_keys in self._query(**keys):
            row = row_keys.copy()
            if load_values:
                row['value'] = self._dict[row_keys_as_str][1]
            yield row
    
    @override
    def delete_all(self, **keys: VALUE):
        for row_keys_as_str, _row_keys in list(self._query(**keys)):
            del self._dict[row_keys_as_str]
    
    def _query(self, **keys: VALUE) -> Iterator[tuple[str, dict[str, Any]]]:
        for row_keys_as_str, (row_keys, _value) in self._dict.items():
            if first_contains_second(row_keys, keys):
                yield row_keys_as_str, row_keys
    
    @override
    def close(self):
        pass


class ShelfStorage(DictStorage):
    """ An implementation of
    :class:`~asr_eval.utils.storage.BaseStorage` based on Python's
    :code:`shelf`.

    With :code:`read_only=True` you can open the same file multiple times
    simultaneously.
    
    Note:
        Methods :meth:`~asr_eval.utils.storage.BaseStorage.list_all` or
        :meth:`~asr_eval.utils.storage.BaseStorage.delete_all` iterate
        over all the rows, which may be slow in this implementation.
    """

    def __init__(self, path: str | Path, read_only: bool = False):
        print(f'Reading from ShelfStorage at {path}...', end=' ')
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        self._dict = shelve.open(str(path), flag='r' if read_only else 'c')
        print(f'read {len(self._dict)} rows')
    
    @override
    def close(self):
        cast(shelve.Shelf[Any], self._dict).close()


class DiskcacheStorage(DictStorage):
    """ An implementation of
    :class:`~asr_eval.utils.storage.BaseStorage` based on diskcache.
    
    Note:
        Methods :meth:`~asr_eval.utils.storage.BaseStorage.list_all` or
        :meth:`~asr_eval.utils.storage.BaseStorage.delete_all` iterate
        over all the rows, which may be slow in this implementation.
    """

    def __init__(self, dir: str | Path):
        from diskcache import Index

        print(f'Reading from diskcache at {dir}...', end=' ')
        Path(dir).parent.mkdir(exist_ok=True, parents=True)
        self._dict = Index(str(dir))
        print(f'read {len(self._dict)} rows')
    
    @override
    def close(self):
        cast(shelve.Shelf[Any], self._dict).close()


def _encode_key(keys: dict[str, VALUE]) -> str:
    """ A conversion that allows to use dict as a mapping's key.
    
    Example:
        >>> print(_encode_key(dict(a='w', c=2, b=True)))
        "[('a', 'w'), ('b', True), ('c', 2)]"
    
    The resulting string can be used as a dict key.
    """
    
    return repr(sorted(keys.items()))


def _is_equal(a: VALUE, b: VALUE) -> bool:
    return type(a) == type(b) and a == b


def first_contains_second(
    full_keys: dict[str, VALUE],
    sub_keys: dict[str, VALUE],
) -> bool:
    try:
        return all(_is_equal(full_keys[k], v) for k, v in sub_keys.items())
    except KeyError:
        return False