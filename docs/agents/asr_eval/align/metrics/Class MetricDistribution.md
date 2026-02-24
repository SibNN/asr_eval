# Class MetricDistribution (defined in asr_eval/align/metrics.py at lines 124-142)

@dataclasses.dataclass
class MetricDistribution:
    """The result of a :func:`~asr_eval.align.metrics.bootstrap`
    algorithm.
    """
    ...

    main_value: float
    """Metric value calculated on the whole dataset."""

    bootstrap_values: list[float]
    """Metric values on bootstrap subsets."""

    def quantiles(self, q: collections.abc.Sequence[float]) -> list[float]:
        """Return the given quantiles of the bootstrap metric
        distribution.
        """
        ...