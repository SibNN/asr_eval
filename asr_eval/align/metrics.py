from __future__ import annotations
from collections.abc import Callable
from functools import partial
from typing import Literal, Self
from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    '''
    TODO docstring
    '''
    true_len: int = 0
    n_replacements: int = 0
    n_insertions: int = 0
    n_deletions: int = 0

    @property
    def n_errors(self) -> int:
        return self.n_replacements + self.n_insertions + self.n_deletions

    def word_error_rate(self, clip: bool = False) -> float:
        wer = self.n_errors / np.clip(self.true_len, 1, None)
        if clip:
            wer = np.clip(wer, 0, 1)
        return wer

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
    

def average_wer(samples: list[Metrics], mode: Literal['plain', 'concat']) -> float:
    '''
    Two example averaging methods (not an exhaustive list of possible methods).
    
    "concat" sums replacements, deletions, insertions and true length for all samples,
    then calculate WER and clip if from 0 to 1.
    
    "plain" clips WER from 0 to 1 for each sample, and then averages for all samples.
    
    If there is a long hallucination (long insertion) in a single sample, this may largely
    affect the overall dataset metric for "concat" mode. This can be mitigated by setting
    max_consecutive_insertions in Alignment.metric_summary().
    
    If true_len > 0 for all samples, the "concat" method is equal to averaging WER
    (with clip=False) for all samples, taking their true_len as averaging weights.
    This is also roughly equal to concatenating all the transcriptions and calculating WER.
    
    Thus, in "concat" method samples with long ground truth transcription have larger effect
    on the overall metric. Also, if we have a long audio without speech, and the count of
    insertions is proportional to the audio length, then such the longer is no-speech audio,
    the more if affects the overall metric.
    
    The "plain" mode is also reasonable, because different samples represent different
    conditions (acoustical, lexical etc.) and can be viewed as many different classes
    (or clusters, or first stage in two-stage sampling). Then, "plain" mode is similar to
    macro-averaging metrics for these classes.
    '''
    match mode:
        case 'plain':
            return float(np.mean([s.word_error_rate(clip=True) for s in samples]))
        case 'concat':
            return sum(samples, start=Metrics()).word_error_rate(clip=True)


@dataclass
class MetricDistribution:
    main_value: float
    bootstrap_values: list[float]
    
    def quantile(self, q: float) -> float:
        return float(np.quantile(self.bootstrap_values, q))


def bootstrap[T](
    samples: list[T],
    calc_metric: Callable[[list[T]], float],
    rounds: int = 100,
    random_seed: int | None = 0,
) -> MetricDistribution:
    rng = np.random.default_rng(seed=random_seed)
    return MetricDistribution(
        main_value=calc_metric(samples),
        bootstrap_values=[
            calc_metric(rng.choice(samples, size=len(samples), replace=True)) # type: ignore
            for _ in range(rounds)
        ]
    )


@dataclass
class DatasetMetric:
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