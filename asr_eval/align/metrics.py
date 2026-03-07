from __future__ import annotations
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal, Self
from dataclasses import dataclass
import base64
import io

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asr_eval.utils.plots import draw_line_with_ticks


__all__ = [
    'Metrics',
    'average_wer',
    'bootstrap',
    'MetricDistribution',
    'DatasetMetric',
    'dataset_metric_to_dataframe',
    'plot_dataset_metric',
]


@dataclass
class Metrics:
    """A dataclass container for error counters to calculate WER (word
    error rate).
    
    To obtain WER value, run
    :meth:`~asr_eval.align.metrics.Metrics.word_error_rate`. See examples
    in the :meth:`~asr_eval.align.alignment.Alignment` docstrings and
    the user guide :doc:`/guide_alignment_wer`.
    """
    
    true_len: int = 0
    n_replacements: int = 0
    n_insertions: int = 0
    n_deletions: int = 0

    @property
    def n_errors(self) -> int:
        """The total number of word errors (replacements + insertions
        + deletions).
        """
        
        return self.n_replacements + self.n_insertions + self.n_deletions

    def word_error_rate(self, clip: bool = False) -> float:
        """The WER (word error rate) value.
        
        If `true_len == 0`, replaces it with 1.
        
        Args:
            clip: If True, the value will be clipped between 0 and 1.
                This helps to stabilize metric, otherwise long
                insertions may lead to a gigantic WER value on a single
                sample, that affects the whole dataset metric and
                depends on the generation limit. See also the related
                parameter :code:`max_consecutive_insertions` in
                :meth:`~asr_eval.align.alignment.Alignment.error_listing`,
                that has similar semantics but is more flexible.
        """
        
        wer = self.n_errors / np.clip(self.true_len, 1, None)
        if clip:
            wer = np.clip(wer, 0, 1)
        return float(wer)
    
    def __radd__(self, other: Metrics) -> Metrics:
        return self.__add__(other)

    def __add__(self, other: Metrics) -> Metrics:
        if isinstance(other, int):
            # for sum(metrics) which is equal sum(metrics, start=0)
            assert other == 0
            other = Metrics()
        return Metrics(
            true_len=self.true_len + other.true_len,
            n_replacements=self.n_replacements + other.n_replacements,
            n_insertions=self.n_insertions + other.n_insertions,
            n_deletions=self.n_deletions + other.n_deletions,
        )
    

def average_wer(
    samples: list[Metrics], mode: Literal['plain', 'concat']
) -> float:
    """Averages WER value for a list of samples.
    
    Two alternative averaging methods are implemented:
    
    1. In "plain" method we calculate WER for each sample, clipping it
    from 0 to 1, and then average all the values.
    2. In "concat" method we sum up each counter (replacements,
    deletions, insertions and ground truth lengths) for all samples, and
    then calculate the WER value from the resulting counters. This is
    roughly equivalent to concatenating all the predictions and ground
    truth before calculating WER. Also, if ground truth length > 0 for
    all samples, this is equal to averaging WER (with
    :code:`clip=False`) for all samples, taking their ground truth
    lengths as averaging weights.
    
    Thus, in "concat" method long samples have larger effect on the
    overall metric, which is reasonable. The "plain" mode is also
    reasonable, because different samples may represent different
    conditions (acoustical, lexical etc.) and can be viewed as many
    different classes (or clusters), and "plain" mode is similar to
    macro-averaging metrics for these classes.
    """
    
    match mode:
        case 'plain':
            return float(np.mean([
                s.word_error_rate(clip=True) for s in samples
            ]))
        case 'concat':
            return sum(samples, start=Metrics()).word_error_rate(clip=True)


@dataclass
class MetricDistribution:
    """The result of a :func:`~asr_eval.align.metrics.bootstrap`
    algorithm.
    """
    
    main_value: float
    """Metric value calculated on the whole dataset."""
    
    bootstrap_values: list[float]
    """Metric values on bootstrap subsets."""
    
    def quantiles(self, q: Sequence[float]) -> list[float]:
        """Return the given quantiles of the bootstrap metric
        distribution.
        """
        
        return np.quantile(self.bootstrap_values, q).tolist()

def bootstrap[T: Sequence[Any] | npt.NDArray[Any]](
    samples: T,
    calc_metric: Callable[[T], float],
    rounds: int = 100,
    random_seed: int | None = 0,
) -> MetricDistribution:
    """Calculates a metric uncertainty via bootstrapping.
    
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
    
    rng = np.random.default_rng(seed=random_seed)
    return MetricDistribution(
        main_value=float(calc_metric(samples)),
        bootstrap_values=[
            calc_metric(rng.choice(samples, size=len(samples), replace=True)) # type: ignore
            for _ in range(rounds)
        ]
    )


@dataclass
class DatasetMetric:
    """Keeps a bootstrap distribution for WER, number of replacements,
    insertions and deletions.
    
    This helps to determine confidence intervals.
    """
    
    wer: MetricDistribution
    n_replacements: MetricDistribution
    n_insertions: MetricDistribution
    n_deletions: MetricDistribution
    
    @classmethod
    def from_samples(
        cls,
        samples: list[Metrics],
        wer_averaging_mode: Literal['plain', 'concat'] = 'concat',
    ) -> Self:
        """Construct bootstrap distributions for the whole dataset from
        individual sample metrics.
        """
        return cls(
            wer=bootstrap(
                samples,
                partial(average_wer, mode=wer_averaging_mode)
            ),
            n_replacements=bootstrap(
                samples,
                lambda samples: sum(samples, start=Metrics()).n_replacements
            ),
            n_insertions=bootstrap(
                samples,
                lambda samples: sum(samples, start=Metrics()).n_insertions
            ),
            n_deletions=bootstrap(
                samples,
                lambda samples: sum(samples, start=Metrics()).n_deletions
            ),
        )
        
    
def dataset_metric_to_dataframe(
    metrics: dict[str, DatasetMetric],
    what: Literal[
        'wer', 'n_replacements', 'n_insertions', 'n_deletions'
    ] = 'wer',
    quantile_1: float = 0.1,
    quantile_2: float = 0.9,
) -> pd.DataFrame:
    """Given bootstrap distributions for several models, summarizes
    them into a table.
    
    A helper function for a dashboard.
    """
    
    df_rows: list[dict[str, Any]] = []
    for name, value in metrics.items():
        bootstrap_distribution: MetricDistribution = getattr(value, what)
        q1, q2 = bootstrap_distribution.quantiles((quantile_1, quantile_2))
        df_rows.append({
            'pipeline': name,
            'value': bootstrap_distribution.main_value,
            'lower': q1,
            'upper': q2,
        })
    return pd.DataFrame(df_rows)
        
    
def plot_dataset_metric(
    metrics: dict[str, DatasetMetric],
    what: Literal[
        'wer', 'n_replacements', 'n_insertions', 'n_deletions'
    ] = 'wer',
    show: bool = True,
    quantile_1: float = 0.1,
    quantile_2: float = 0.9,
) -> str:
    """Given bootstrap distributions for several models, summarizes
    them into a plot.
    
    A helper function for a dashboard.
    
    If :code:`show=True`, calls :code:`plt.show()` afterwards. Returns
    base64-encoded image.
    """
    
    Y_DELTA = 0.3
    Y_PAD = 0.2
    Y_TICK = 0.1
    fig_height = Y_PAD * 2 + Y_DELTA * (len(metrics) - 1)

    plt.figure(figsize=(5, fig_height)) # type: ignore
    ax = plt.gca()
    
    plt.title(what) # type: ignore
    
    for i, value in enumerate(metrics.values()):
        bootstrap_distribution: MetricDistribution = getattr(value, what)
        q1, q2 = bootstrap_distribution.quantiles((quantile_1, quantile_2))
        draw_line_with_ticks(
            q1, q2, i, y_tick_width=Y_TICK / Y_DELTA, ax=ax, color='C0'
        )
        plt.scatter( # type: ignore
            [bootstrap_distribution.main_value], [i],
            s=40, marker='|', color='C0'
        )
    
    ax.set_yticks(range(len(metrics))) # type: ignore
    ax.set_yticklabels(list(metrics)) # type: ignore
    
    ax.set_ylim(-Y_PAD / Y_DELTA, len(metrics) - 1 + Y_PAD / Y_DELTA)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='svg', bbox_inches='tight') # type: ignore
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode()

    if show:
        plt.show() # type: ignore
    
    return encoded_image