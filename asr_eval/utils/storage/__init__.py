from pathlib import Path
from asr_eval.utils.storage.base_storage import VALUE, BaseStorage
from asr_eval.utils.storage.dict_storage import (
    DictStorage, ShelfStorage, DiskcacheStorage
)
from asr_eval.utils.storage.csv_storage import CSVStorage


__all__ = [
    'VALUE',
    'BaseStorage',
    'DictStorage',
    'ShelfStorage',
    'CSVStorage',
    'DiskcacheStorage',
]


# def get_storage_class(name: str) -> type[BaseStorage]:
#     """Gets class by name, such as "BaseStorage" -> `BaseStorage`.

#     If not found, tries to get from the full path, e.g.
#     :code:`get_storage_class("my_package.module.MyStorage")`.

#     Raises :code:`ImportError` if cannot find the specified class.
#     """
#     if '.' not in name:
#         name = f'asr_eval.utils.storage.{name}'
#     from hydra.utils import get_object
#     return get_object(name)


def make_storage(path: str | Path | None) -> BaseStorage:
    """An utility to make a storage for command line tools.
    
    - Use .csv path to instantiate
      :class:`~asr_eval.utils.storage.CSVStorage`
    - Use dir name (without dot in name) to instantiate
      :class:`~asr_eval.utils.storage.DiskcacheStorage` (binary and more
      efficient).
    - Use .dbm path to instantiate
      :class:`~asr_eval.utils.storage.ShelfStorage` (binary and more
      efficient).
    - Use :code:`None` to instantiate in-memory
      :class:`~asr_eval.utils.storage.DictStorage`.
    """

    if path is None:
        return DictStorage()
    path = Path(path)
    if path.suffix == '.csv':
        return CSVStorage(path)
    elif len(path.suffix):
        return ShelfStorage(path)
    else:
        return DiskcacheStorage(path)