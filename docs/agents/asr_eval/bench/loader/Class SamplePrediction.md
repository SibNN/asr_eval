# Class SamplePrediction (defined in asr_eval/bench/loader.py at lines 41-48)

@dataclasses.dataclass(frozen=True)
class SamplePrediction:
    """A value to group predictions in
    :class:`~asr_eval.bench.loader.PredictionLoader`.
    """
    ...

    text: str

    elapsed_time: float