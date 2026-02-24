# Class DatasetInfo (defined in asr_eval/bench/datasets/_registry.py at lines 107-118)

@dataclasses.dataclass
class DatasetInfo:
    """A container for dataset information that is stored if a dataset
    gets registered.
    """
    ...

    instantiate_fn: typing.Callable[[str], Dataset]

    splits: tuple[str, ...]

    unlabeled: bool

    filter: typing.Callable[[str], list[int]] | None = None