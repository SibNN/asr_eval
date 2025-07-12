from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


AUDIO_TYPE = npt.NDArray[np.floating[Any]]

class ASREvalWrapper(ABC):
    @abstractmethod
    def __call__(self, waveforms: list[AUDIO_TYPE]) -> list[str]:
        ...