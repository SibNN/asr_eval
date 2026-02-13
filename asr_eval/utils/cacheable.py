from collections.abc import Callable
from pathlib import Path
import shelve
import dbm


class DiskCacheable:
    """A wrapper for callable that converts string to string. Caches the
    inputs and outputs into a file using Python's :code:`shelf`.
    """
    
    shelf: shelve.Shelf[str] | dict[str, str]
    
    def __init__(
        self,
        fn: Callable[[str], str],
        cache_path: str | Path,
    ):
        self.fn = fn
        try:
            self.shelf = shelve.open(str(cache_path))
        except dbm.error as e:
            print(
                f'Cannot use disk cache, will use in-memory cache; error: {e}'
            )
            self.shelf = {}
    
    def __call__(self, text: str) -> str:
        if (cached := self.shelf.get(text, None)):
            return cached
        self.shelf[text] = result = self.fn(text)
        return result