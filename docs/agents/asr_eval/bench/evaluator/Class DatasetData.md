# Class DatasetData (defined in asr_eval/bench/evaluator.py at lines 30-59)

@dataclasses.dataclass
class DatasetData:
    """Output format for the
    :class:`~asr_eval.bench.evaluator.get_dataset_data` function.
    """
    ...

    samples: list[asr_eval.bench.evaluator.SampleData]
    """A list of :class:`~asr_eval.bench.evaluator.SampleData` for
    all the sample ids for which we have at least one prediction.
    """

    full_samples: list[int]
    """A list of sample ids for which all the pipelines have a
    prediction. These sample ids are used for averaging metrics, to
    avoid a problem where different pipeline predicions are averaged
    across differen samples set, and hence are not directly comparable.
    """

    dataset_metric: dict[str, asr_eval.align.metrics.DatasetMetric]
    """Metrics for each pipeline, averaged across
    :attr:`~asr_eval.bench.evaluator.DatasetData.full_samples`,
    if the :code:`full_samples` list is not empty.
    """

    def get_all_pipelines(self) -> list[str]:
        """Get all the pipelines for which we have at least one
        prediction.
        """
        ...