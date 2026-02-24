# Function bootstrap (defined in asr_eval/align/metrics.py at lines 143-182)

def bootstrap[T: Sequence[Any] | npt.NDArray[Any]](
    samples: T,
    calc_metric: collections.abc.Callable[[T], float],
    rounds: int = 100,
    random_seed: int | None = 0,
) -> asr_eval.align.metrics.MetricDistribution:
    """Calculate a metric uncertainty via boostrapping.

    Given a list of samples and a function :code:`calc_metric` that
    calcualtes averaged metric, run the function :code:`rounds` times,
    each time selecting N from N samples with replacement, with the
    given `random_seed`. Returns the results in the
    :attr:`~asr_eval.align.metrics.MetricDistribution.bootstrap_values`
    field.

    Also applies the :code:`calc_metric` to the whole :code:`samples`
    list without subsampling and returns the result in the
    :attr:`~asr_eval.align.metrics.MetricDistribution.main_value`
    field.

    Example:
        >>> import numpy as np
        >>> from asr_eval.align.metrics import bootstrap
        >>> outcomes = np.random.default_rng(0).integers(0, 2, size=100)
        >>> distribution = bootstrap(outcomes, np.mean)
        >>> distribution.quantiles((0.1, 0.9))
        [0.509, 0.6310000000000001]
        >>> distribution.main_value
        0.56
    """
    ...