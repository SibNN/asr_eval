# Class Metrics (defined in asr_eval/align/metrics.py at lines 28-87)

@dataclasses.dataclass
class Metrics:
    """A dataclass container for error counters to calculate WER (word
    error rate).

    To obtain WER value, run
    :meth:`~asr_eval.align.metrics.Metrics.word_error_rate`. See examples
    in the :meth:`~asr_eval.align.alignment.Alignment` dostrings and
    the user guide :doc:`/guide_alignment_wer`.
    """
    ...

    true_len: int = 0

    n_replacements: int = 0

    n_insertions: int = 0

    n_deletions: int = 0

    @property
    def n_errors(self) -> int:
        """The total number of word errors (replacements + insertions
        + deletions).
        """
        ...

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
                that have a similar semantic but is more flexible.
        """
        ...