# Function rolling_window (defined in asr_eval/utils/misc.py at lines 57-71)

def rolling_window[T: (INTS, FLOATS)](arr: T, size: int) -> T:
    """Returns all subarrays of length :code:`size`, stacked together
    along a new axis.

    Example:
        >>> rolling_window(np.array([1, 0, 2, 1, 3, 5]), 3) # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 2], [0, 2, 1], [2, 1, 3], [1, 3, 5]])

    Taken from: https://stackoverflow.com/a/7100681
    """
    ...