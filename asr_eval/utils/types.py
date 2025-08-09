from typing import Any
import numpy as np
import numpy.typing as npt


__all__ = [
    "FLOATS",
    "INTS",
]


FLOATS = npt.NDArray[np.floating[Any]]
INTS = npt.NDArray[np.integer[Any]]