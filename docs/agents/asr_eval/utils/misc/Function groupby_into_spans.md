# Function groupby_into_spans (defined in asr_eval/utils/misc.py at lines 25-39)

def groupby_into_spans[T](
    iterable: collections.abc.Iterable[T]
) -> collections.abc.Iterable[tuple[T, int, int]]:
    """Find spans of the same value in a sequence. Returns (value,
    start_index, end_index).

    Example:
        >>> list(groupby_into_spans(['x', 'x', 'b', 'a', 'a', 'a']))
        [('x', 0, 2), ('b', 2, 3), ('a', 3, 6)]
    """
    ...