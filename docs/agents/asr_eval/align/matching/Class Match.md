# Class Match (defined in asr_eval/align/matching.py at lines 34-83)

@dataclasses.dataclass(kw_only=True, slots=True)
class Match:
    """A dataclass for a single match between words when aligning a
    pair of texts.

    Note:
        This is a lower-level object only needed if you work with
        :func:`~asr_eval.align.solvers.dynprog.Match.solve_optimal_alignment`
        directly. If you work with
        :func:`~asr_eval.align.alignment.Alignment`, matches are
        automatically converted into
        :attr:`~asr_eval.align.alignment.Alignment.slots`, so you
        don't operate with them directly.
    """
    ...

    true: asr_eval.align.transcription.Token | None
    """A word from the first text."""

    pred: asr_eval.align.transcription.Token | None
    """A word from the second text."""

    status: typing.Literal['correct', 'deletion', 'insertion', 'replacement']
    """One of 4 possible statuses that are standard for the string
    matching problem:

    - If "correct" or "replacement", both tokens are not None. The match
      is between some token in the ground truth and some token in the
      prediction, and they are either equal ("correct") or not equal
      ("replacement").
    - If "deletion", the pred token is None. This match represents a
      token existing in the ground truth but not existing in the
      prediction.
    - If "insertion", the true token is None. This match represents a
      token existing in the prediction but not existing in the
      ground truth.
    """

    score: asr_eval.align.matching.AlignmentScore
    """An associated alignment score for the current match.

    Roughly, it keeps 0 or 1, depending on whether the words match or
    not. If the `true` word is a
    :class:`~asr_eval.align.transcription.Wildcard`, then the alignment
    score is also 0, because wildcard matches with anything.
    """