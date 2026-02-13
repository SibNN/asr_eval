from typing import Any
import numpy as np
import numpy.typing as npt


__all__ = [
    "FLOATS",
    "INTS",
]


FLOATS = npt.NDArray[np.floating[Any]]
"""A typization for numpy array of floats"""

INTS = npt.NDArray[np.integer[Any]]
"""A typization for numpy array of integers"""