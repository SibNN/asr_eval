from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Literal, cast
import os

import nltk

from asr_eval.align.transcription import Wildcard, Token, MultiVariantBlock


__all__ = [
    'AlignmentScore',
    'Match',
    'MatchesList',
    'char_edit_distance',
]


DISABLE_BETTER_ALIGNMENT = bool(int(
    os.environ.get('ASR_EVAL_DISABLE_BETTER_ALIGNMENT', 0)
))


@cache
def char_edit_distance(true: str, pred: str) -> int:
    """A :code:`@cache` wrapper for `nltk.edit_distance`. Calculates
    character edit distance between strings.
    """
    return nltk.edit_distance(true, pred) # type: ignore


@dataclass(kw_only=True, slots=True)
class Match:
    """A dataclass for a single match between words when aligning a
    pair of texts.
    
    Note:
        This is a lower-level object only needed if you work with
        :func:`~asr_eval.align.solvers.dynprog.solve_optimal_alignment`
        directly. If you work with
        :func:`~asr_eval.align.alignment.Alignment`, matches are
        automatically converted into
        :attr:`~asr_eval.align.alignment.Alignment.slots`, so you
        don't operate with them directly.
    """
    true: Token | None
    """A word from the first text."""
    
    pred: Token | None
    """A word from the second text."""
    
    status: Literal['correct', 'deletion', 'insertion', 'replacement']
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
    
    score: AlignmentScore
    """An associated alignment score for the current match.
    
    Roughly, it keeps 0 or 1, depending on whether the words match or
    not. If the `true` word is a
    :class:`~asr_eval.align.transcription.Wildcard`, then the alignment
    score is also 0, because wildcard matches with anything.
    """

    def __repr__(self) -> str:
        first = str(self.true.value) if self.true is not None else ''
        second = str(self.pred.value) if self.pred is not None else ''
        return f'Match({first}, {second})'


def _match_from_pair(true: Token | None, pred: Token | None) -> Match: # pyright: ignore[reportUnusedFunction]
    """
    Constructs `Match` object and fills `status` and `score` fields.
    
    This function does not solve an optimal alighment problem. If both
    tokens are the same, or the first is :code:`Wildcard()`, then
    match is considered 'correct', otherwise incorrect. In incorrect
    match, if both texts are not empty, it is considered 'replacement',
    otherwise 'deletion' or 'insertion'.
    
    This is a helper function that is used during solving an optimal
    alignment problem.
    """
    T = true is not None
    P = pred is not None
    assert T or P
    
    is_anything = T and isinstance(true.value, Wildcard)
    if is_anything or (T and P and true.value == pred.value):
        status = 'correct'
    elif T and P:
        status = 'replacement'
    elif T:
        status = 'deletion'
    else:
        status = 'insertion'
    
    return Match(
        true=true,
        pred=pred,
        status=status,
        score=AlignmentScore(
            n_word_errors=0 if status == 'correct' else 1,
            n_correct=int(T) if status == 'correct' and not is_anything else 0,
            n_char_errors=char_edit_distance(
                str(true.value) if T else '',
                str(pred.value) if P else '',
            ) if (not is_anything and status != 'correct') else 0,
        ),
    )


@dataclass(slots=True)
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
    
    n_word_errors: int = 0
    """The total number of word errors (replacements + deletions +
    insertions).
    """
    
    n_correct: int = 0
    """The total number of correct matches. Consider the case where
    "so nothing" matches with "nothing huh". We can match "so" with
    "nothing" and "nothing" with "huh" - this gives n_word_errors = 2
    that is optimal. Alternatively, we can match "nothing" with
    "nothing", and let "so" be deletion and "huh" be insertion. This
    also gives n_word_errors = 2, but is clearly better.
    """
    
    n_char_errors: int = 0
    """The sum of character errors in each matches. Note that this is
    not related CER, because if we match "no thing" with "nothing" we
    get n_char_errors = 2 + 2 ("no" deletion plus "thing" to "nothing"
    replacement). This is larger than number of errors in character
    alignment (which is 1).
    """

    def __add__(self, other: AlignmentScore) -> AlignmentScore:
        # score for a concatenation 
        return AlignmentScore(
            self.n_word_errors + other.n_word_errors,
            self.n_correct + other.n_correct,
            self.n_char_errors + other.n_char_errors,
        )

    def _compare(self, other: AlignmentScore) -> Literal['<', '=', '>']:
        # comparison order:

        # 1. n_word_errors (lower is better)
        if self.n_word_errors > other.n_word_errors:
            return '<'
        if self.n_word_errors < other.n_word_errors:
            return '>'
        
        if DISABLE_BETTER_ALIGNMENT:
            return '='

        # 2. n_char_errors (lower is better)
        if self.n_char_errors > other.n_char_errors:
            return '<'
        if self.n_char_errors < other.n_char_errors:
            return '>'

        # 3. n_correct (higher is better)
        if self.n_correct < other.n_correct:
            return '<'
        if self.n_correct > other.n_correct:
            return '>'

        return '='

    # do not use functools.total_ordering to speedup

    def __lt__(self, other: object) -> bool:
        return self._compare(cast(AlignmentScore, other)) == '<'

    def __gt__(self, other: object) -> bool:
        return self._compare(cast(AlignmentScore, other)) == '>'

    def __eq__(self, other: object) -> bool:
        return self._compare(cast(AlignmentScore, other)) == '='

    def __ne__(self, other: object) -> bool:
        return self._compare(cast(AlignmentScore, other)) != '='

    def __le__(self, other: object) -> bool:
        return self._compare(cast(AlignmentScore, other)) != '>'

    def __ge__(self, other: object) -> bool:
        return self._compare(cast(AlignmentScore, other)) != '<'


def _get_len(tokens: Sequence[Token]) -> int:
    return len([t for t in tokens if not isinstance(t.value, Wildcard)])
    

def _select_shortest_multi_variants( # pyright: ignore[reportUnusedFunction]
    seq: Sequence[Token | MultiVariantBlock]
) -> tuple[Token, ...]:
    """Selects the shortest option in each multivariant block."""
    result: list[Token] = []
    for x in seq:
        if isinstance(x, MultiVariantBlock):
            result += min(x.options, key=_get_len)
        else:
            result.append(x)
    return tuple(result)


def _true_len(match: Match) -> int:
    if match.true is not None and isinstance(match.true.value, Wildcard):
        # Wildcard blocks do not increment true len!
        return 0
    return int(match.true is not None)


@dataclass(slots=True)
class MatchesList:
    """The result of the optimal alignment algorithm.
    """
    matches: list[Match]
    """A list of matches (correct, replacements, deletions or
    insertions) that together form an optimal alignment."""
    
    total_true_len: int
    """A total length of the ground truth.
    
    If there are multivariant blocks in the ground truth, only the
    selected block (the one that matched with the prediction)
    contributes to the `total_true_len`.
    
    Also, :class:`~asr_eval.align.transcription.Wildcard` tokens in the
    ground truth do not increment the total_true_len. See also
    :meth:`asr_eval.align.alignment.Alignment.get_true_len`.
    """
    
    score: AlignmentScore
    """A total alignment score. Contains word error counts and some
    other metrics we try to optimize."""

    @classmethod
    def from_list(cls, matches: list[Match]) -> MatchesList:
        """An internal method to construct.
        
        :meta private:
        """
        return MatchesList(
            matches=matches,
            total_true_len=sum(_true_len(m) for m in matches),
            score=sum((m.score for m in matches), AlignmentScore())
        )

    def prepend(self, match: Match) -> MatchesList:
        """An internal method to extend left.
        
        :meta private:
        """
        # ! This is O(n) because it copies the entire list
        return MatchesList(
            matches=[match] + self.matches,
            total_true_len=_true_len(match) + self.total_true_len,
            score = match.score + self.score
        )

    def append(self, match: Match) -> MatchesList:
        """An internal method to extend right.
        
        :meta private:
        """
        return MatchesList(
            matches=self.matches + [match],
            total_true_len=self.total_true_len + _true_len(match),
            score = self.score + match.score
        )