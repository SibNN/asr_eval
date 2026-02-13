from __future__ import annotations
import copy
from enum import Enum
from functools import cache
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, cast
from abc import ABC, abstractmethod
from dataclasses import fields, _is_dataclass_instance, _ATOMIC_TYPES # type: ignore

# from hydra.utils import log as _log
# _log.setLevel('FATAL')  # for get_obj messages on ImportError


__all__ = [
    'SerializableToDict',
    'save_to_json',
    'load_from_json',
    'serialize_object',
    'deserialize_object',
]


class SerializableToDict(ABC):
    """An interface to to serialize an object into a json-compatibl
    dict with
    :func:`~asr_eval.utils.serializing.serialize_object` and load back
    with :func:`~asr_eval.utils.serializing.deserialize_object`.
    
    Is not needed for dataclasses, only for objects with custom
    (de)serialization logic.
    """
    @abstractmethod
    def serialize_to_dict(self) -> dict[str, Any]:
        """Returns a dict to write into json. The resulting dict can be
        passed back into the class constructor to restore an equal
        object.
        """


def save_to_json(obj: Any, path: str | Path, indent: int = 4):
    """Serializes an hierarchical structure of dataclasses/lists/dicts
    to a json-compatible dict and then saves to a .json file. Can be
    loaded back with :func:`~asr_eval.utils.serializing.load_from_json`.
    
    If an exception or keyboard interrupt happens during saving, the
    file will not br created.
    """
    try:
        import hydra_zen # pyright: ignore[reportUnusedImport]
    except ImportError:
        raise ImportError('Install hydra_zen to use `save_to_json`')
    
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(
        mode='w', delete=True, dir='.'
    ) as temp_file_info:
        temp_path = Path(temp_file_info.name)
        temp_path.write_text(
            json.dumps(
                serialize_object(obj), indent=indent, ensure_ascii=False
            ),
            encoding='utf-8',
        )
        temp_path.rename(path)


def load_from_json(path: str | Path) -> Any:
    """Loads a data structure that was saved with
    :func:`~asr_eval.utils.serializing.save_to_json`. If the .json file
    does not contain any `_target_` fields, will act equally to
    :code:`json.loads(path.read_text())`.
    """
    
    try:
        import hydra_zen # pyright: ignore[reportUnusedImport]
    except ImportError:
        raise ImportError('Install hydra_zen to use `load_from_json`')
    
    return deserialize_object(json.loads(Path(path).read_text()))


def _get_obj_path(obj: Any) -> str:
    from hydra_zen.structured_configs._implementations import DefaultBuilds
    return DefaultBuilds()._get_obj_path(obj.__class__) # pyright: ignore[reportPrivateUsage]


@cache
def _get_obj_from_path(target: str) -> type | Callable[..., Any]:
    from hydra_zen.funcs import get_obj
    return get_obj(path=target)


def serialize_object(obj: Any) -> Any:
    """ Serializes an hierarchical structure of dataclasses, lists,
    dicts or enums into a json-compatible dict.
    
    This includes converting dataclasses into dicts (omitting fields
    where the value is None and the default value is also None). The
    class full name is written to the additional :code:`_target_`
    field to construct the object back with
    :func:`~asr_eval.utils.serializing.deserialize_object`.
    
    Besides dataclasses, can serialize
    :func:`~asr_eval.utils.serializing.SerializableToDict` objects. This
    is useful for custom classes that are not dataclasses, but we want
    to be able to save (to json or yaml) and load them.
    """
    
    result_dict: dict[str, Any]
    if isinstance(obj, SerializableToDict):
        result_dict = {'_target_': _get_obj_path(obj)}
        result_dict.update(obj.serialize_to_dict())
        return result_dict
    elif _is_dataclass_instance(obj):
        result_dict = {'_target_': _get_obj_path(obj)}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if f.default is None and value is None:
                continue
            result_dict[f.name] = serialize_object(value)
        return result_dict
    elif isinstance(obj, Path):
        result_dict = {'_target_': _get_obj_path(obj)}
        result_dict['*args'] = [str(obj)]
        return result_dict
    elif isinstance(obj, Enum):
        return {
            '_enum_': _get_obj_path(obj),
            '_name_': obj.name,
        }
    elif type(obj) in _ATOMIC_TYPES:
        return obj
    elif type(obj) == list:
        return [serialize_object(x) for x in obj] # type: ignore
    elif type(obj) == dict:
        result_dict = {}
        for k, v in obj.items(): # type: ignore
            if not type(k) in _ATOMIC_TYPES: # type: ignore
                raise NotImplementedError(
                    'Can serialize only dicts'
                    f' with atomic keys, got {type(k)}' # type: ignore
                )
            result_dict[k] = serialize_object(v)
        return result_dict
    else:
        raise NotImplementedError(
            'Can serialize only primitives, lists, dicts, dataclasses'
            f' and SerializableToDict, cannot serialize {obj!r}'
        )


def deserialize_object(serialized: Any, ignore_errors: bool = False) -> Any:
    """Deserializes an object serialized with
    :func:`~asr_eval.utils.serializing.serialize_object`.
    
    If no :code:'_target_' fields found, returns the input data without
    changes.
    """
    try:
        serialized = copy.deepcopy(serialized)
        if type(serialized) in _ATOMIC_TYPES:
            return serialized
        elif type(serialized) == list:
            return [deserialize_object(x) for x in serialized] # type: ignore
        elif type(serialized) == dict:
            if '_enum_' in serialized:
                enum_cls = _get_obj_from_path(cast(str, serialized['_enum_']))
                return getattr(enum_cls, cast(str, serialized['_name_']))
            target = serialized.pop('_target_', None) # type: ignore
            content = {k: deserialize_object(v) for k, v in serialized.items()} # type: ignore
            if target is not None:
                assert isinstance(target, str)
                try:
                    cls = _get_obj_from_path(target)
                except ImportError as e:
                    if not ignore_errors:
                        raise e
                    print(
                        'Ignoring error while deserializing:'
                        f' cannot import {target}'
                    )
                    content['_target_'] = target
                    return content # type: ignore
                args = content.pop('*args', []) # type: ignore
                return cls(*args, **content)
            else:
                return content # type: ignore
        pass
    except Exception as e:
        if not ignore_errors:
            raise e
        else:
            print(f'Ignoring error while deserializing: {e}')
            return serialized # type: ignore