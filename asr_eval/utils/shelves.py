from pathlib import Path
import shelve
from collections.abc import Iterator, MutableMapping
from typing import Any, Self


class TupleKeyShelf(MutableMapping[tuple[str, ...], Any]):
    """A wrapper around a :code:`shelve.Shelf` that uses tuples of
    strings as keys. Internally, keys are stored as a single string
    joined by NUL (\0) to avoid collisions.
    """
    
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        self._shelf = shelve.open(str(path))
    
    def _encode_key(self, key: tuple[str, ...]) -> str:
        return '\0'.join(key)
    
    def _decode_key(self, key_str: str) -> tuple[str, ...]:
        return tuple(key_str.split('\0'))
    
    def __getitem__(self, key: tuple[str, ...]) -> Any:
        return self._shelf[self._encode_key(key)]
    
    def __setitem__(self, key: tuple[str, ...], value: Any):
        self._shelf[self._encode_key(key)] = value
    
    def __delitem__(self, key: tuple[str, ...]):
        del self._shelf[self._encode_key(key)]
    
    def __iter__(self) -> Iterator[tuple[str, ...]]:
        for key_str in self._shelf:
            yield self._decode_key(key_str)
    
    def __len__(self) -> int:
        return len(self._shelf)
    
    def close(self) -> None:
        self._shelf.close()
    
    def sync(self) -> None:
        self._shelf.sync()
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): # type: ignore
        self.close()
    
    # def get_or_write(self, key: tuple[str, ...], fn: Callable[[], Any]) -> Any:
    #     '''
    #     Similar to .setdefault(key, obj), but accepts a function. It returns key
    #     if exists, otherwise fills the key with fn() and return the result.
    #     '''
    #     try:
    #         return self[key]
    #     except KeyError:
    #         print('calculating', key)
    #         result = self[key] = fn()
    #         return result