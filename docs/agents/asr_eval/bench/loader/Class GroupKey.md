# Class GroupKey (defined in asr_eval/bench/loader.py at lines 31-40)

@dataclasses.dataclass(frozen=True)
class GroupKey:
    """A key to group predictions in
    :class:`~asr_eval.bench.loader.PredictionLoader`.
    """
    ...

    pipeline_name: str

    dataset_name: str

    augmentor: str

    parser: str