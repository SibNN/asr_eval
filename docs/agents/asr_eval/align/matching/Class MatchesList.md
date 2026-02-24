# Class MatchesList (defined in asr_eval/align/matching.py at lines 243-300)

@dataclasses.dataclass(slots=True)
class MatchesList:
    """The result of the optimal alignment algorithm.
    """
    ...

    matches: list[asr_eval.align.matching.Match]
    """A list of matches (correct, replacements, deletions or
    insertions) that together form an optimal alignment."""

    total_true_len: int
    """A total length of the ground truth.

    If there are multivariant blocks in the ground truth, only the
    selected block (the one that matched with the prediction) contribute
    to the `total_true_len`.

    Also, :class:`~asr_eval.align.transcription.Wildcard` tokens in the
    ground truth do not increment the total_true_len. See also
    :meth:`asr_eval.align.alignment.Alignment.get_true_len`.
    """

    score: asr_eval.align.matching.AlignmentScore
    """A total alignment score. Contains word error counts and some
    other metrics we try to optimize."""

    @classmethod
    def from_list(cls, matches: list[asr_eval.align.matching.Match]) -> asr_eval.align.matching.MatchesList:
        """An internal method to construct.

        :meta private:
        """
        ...

    def prepend(self, match: asr_eval.align.matching.Match) -> asr_eval.align.matching.MatchesList:
        """An internal method to extend left.

        :meta private:
        """
        ...

    def append(self, match: asr_eval.align.matching.Match) -> asr_eval.align.matching.MatchesList:
        """An internal method to extend right.

        :meta private:
        """
        ...