# Class DatasetMetric (defined in asr_eval/align/metrics.py at lines 184-224)

@dataclasses.dataclass
class DatasetMetric:
    """Keeps a bootstrap distribution for WER, number of replacements,
    insertions and deletions.

    This helps to determine confidence intervals.
    """
    ...

    wer: asr_eval.align.metrics.MetricDistribution

    n_replacements: asr_eval.align.metrics.MetricDistribution

    n_insertions: asr_eval.align.metrics.MetricDistribution

    n_deletions: asr_eval.align.metrics.MetricDistribution

    @classmethod
    def from_samples(
        cls,
        samples: list[asr_eval.align.metrics.Metrics],
        wer_averaging_mode: typing.Literal['plain', 'concat'] = 'concat',
    ) -> typing.Self:
        """Construct bootstrap distributions for the whole dataset from
        individual sample metrics.
        """
        ...