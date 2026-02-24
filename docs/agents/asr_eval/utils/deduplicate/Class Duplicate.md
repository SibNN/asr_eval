# Class Duplicate (defined in asr_eval/utils/deduplicate.py at lines 123-142)

@dataclasses.dataclass
class Duplicate:
    """An information about found duplicate."""
    ...

    mode: typing.Literal['whole', 'partial']
    """If "partial" this is a duplicate with different slicing. For
    example, if sample #1 has a length of 10 seconds, and sample #0 is
    a slice of sample #1 from 3 to 7 seconds, then they both form a
    :code:`Duplicate(mode='partial', sample_idxs=[0, 1])`.
    """

    sample_idxs: list[int]
    """A list of sample indices that are considered duplicates."""