from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asr_eval.bench.datasets._registry import AudioSample


augmentors_registry: dict[str, type[AudioAugmentor]] = {}
"""A registry where the registered augmentors are stored."""

augmentors_instances: dict[str, AudioAugmentor] = {}


class AudioAugmentor(ABC):
    """Abstract audio preprocessor, primarily for evaluation with
    artificial noises.
    
    To register an augmentor, one need to subclass this class and define
    the :code:`__call__` method that processes an audio sample.

    Preferrably should not modify the input dict and return a copy.
    
    TODO example.
    """
    @abstractmethod
    def __call__(self, sample: AudioSample) -> AudioSample: ...
    
    def __init_subclass__(cls, register_as: str | None = None, **kwargs: Any):
        global augmentors_registry
        super().__init_subclass__(**kwargs)
        if register_as:
            assert register_as not in augmentors_registry
            augmentors_registry[register_as] = cls
            cls.name = register_as


def get_augmentor(name: str):
    """Retrieve a registered augmentor. Will instantiate this augmentor
    and return it on all subsequent calls.
    """
    assert name in augmentors_registry, f'Augmentor {name} is not registered'
    if name in augmentors_instances:
        return augmentors_instances[name]
    augmentors_instances[name] = augmentor = augmentors_registry[name]()
    return augmentor


def unload_augmentor(name: str):
    assert name in augmentors_registry, f'Augmentor {name} is not registered'
    augmentors_instances.pop(name, None)