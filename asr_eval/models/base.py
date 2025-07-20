from abc import ABC, abstractmethod

from ..utils.types import FLOATS


class ASREvalWrapper(ABC):
    @abstractmethod
    def __call__(self, waveforms: list[FLOATS]) -> list[str]:
        ...