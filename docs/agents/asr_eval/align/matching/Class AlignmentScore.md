# Class AlignmentScore (defined in asr_eval/align/matching.py at lines 127-217)

@dataclasses.dataclass(slots=True)
class AlignmentScore:
    """A joint score that we try to optimize during optimal alignment.

    Keeps 3 metrics that we compare lexicographically: compare the
    first, if equal compare the second, and if also equal compare
    the third. This ensures that we always find an alignment that is
    optimal by
    :attr:`~asr_eval.align.matching.AlignmentScore.n_word_errors`, but
    also may be good by other two metrics. This helps to improve
    alignments, that is especially important for streaming recognition,
    because to evaluate latency we need to obtain a good alignment, not
    only the WER value.
    """
    ...

    n_word_errors: int = 0
    """The total number of word errors (replacements + deletions +
    insertions).
    """

    n_correct: int = 0
    """The total number of correct matches. Consider the case where
    "so nothing" matches with "nothing huh". We can match "so" with
    "nothing" and "nothing" with "huh" - this gives n_word_errors = 2
    that is optimal. Alternatively, we can match "nothing" with
    "nothing", and let "so" be deletion and "huh: be insertion. This
    also gives n_word_errors = 2, but is clearly better.
    """

    n_char_errors: int = 0
    """The sum of character errors in each matches. Note that this is
    not related CER, because if we match "no thing" with "nothing" we
    get n_char_errors = 2 + 2 ("no" deletion plus "thing" to "nothing"
    replacement). This is larger than number of errors in character
    alignment (which is 1).
    """