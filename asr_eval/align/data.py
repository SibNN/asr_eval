from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from asr_eval.utils.misc import new_uid


@dataclass(slots=True)
class Anything:
    """
    Represents <*> in a multivariant transcription: a symbol that matches every word sequence or nothing.
    """
    def __eq__(self, other: Any) -> bool:
        return True
    
    # def __repr__(self) -> str:
    #     return '<*>'


@dataclass(slots=True)
class Token:
    """
    Either a word, or `Anything` in a transcription. Additional fields:
    
    - `uid`: a unique id for the current token (is filled automatically if not specified). Useful in
    string alignments, because there can be multiple equal words in both texts, and without unique IDs
    the alignment will be ambiguous. We could use positions instead of unqiue IDs, but positions are
    also ambiguous in a multivariant transcription.
    - `start_pos` and `end_pos`: position in the original text: start (inclusive) and end (exclusive)
    characters. May be useful for displaying an alignment.
    - `start_time` and `end_time`: start and end time in seconds, if known.
    - `type` is either "word", or "punct" (see `split_text_into_tokens`), or any user-defined types
    """
    value: str | Anything
    uid: str = field(default_factory=new_uid)
    start_pos: int = 0
    end_pos: int = 0
    start_time: float = np.nan
    end_time: float = np.nan
    type: str = 'word'
    
    def __repr__(self) -> str:
        strings = [
            str(self.value),
            # f'pos=({self.start_pos}, {self.end_pos})'
        ]
        if self.is_timed:
            strings.append(f't=({self.start_time:.1f}, {self.end_time:.1f})')
        if self.type != 'word':
            strings.append(self.type)
        return f'Token(' + ', '.join(strings) + ')'

    @property
    def is_timed(self) -> bool:
        return not np.isnan(self.start_time) and not np.isnan(self.end_time)


@dataclass(slots=True)
class MultiVariant:
    """
    A multivariant block in a transcription. Keeps a list of variants, each variant is a list of Token.
    
    `pos` represents a position in the original text, including braces {}.
    """
    options: list[list[Token]]
    pos: tuple[int, int] = (0, 0)
    
    def __repr__(self) -> str:
        return f'MultiVariant({str(self.options)[1:-1]})'

    @property
    def is_timed(self) -> bool:
        return all(all(t.is_timed for t in option) for option in self.options)
    
    @property
    def start_time(self) -> float:
        """
        Will return the earliest .start_time across all options, or NaN if tokens are not timed
        """
        start_times = [option[0].start_time for option in self.options if len(option)]
        assert len(start_times), 'All options are empty in a MultiVariant block, should not happen'
        return np.min(start_times)  # `min` builtin works incorrectly: min(-1.0, np.nan) --> -1.0
    
    @property
    def end_time(self) -> float:
        """
        Will return the latest .end_time across all options, or NaN if tokens are not timed
        """
        end_times = [option[-1].end_time for option in self.options if len(option)]
        assert len(end_times), 'All options are empty in a MultiVariant block, should not happen'
        return np.max(end_times)


@dataclass(slots=True)
class AlignmentScore:
    """
    A score to compare for one or more consecutive Match. Comparision algorithm:
    1. First, if one match has less word errors than other, it is better.
    2. Else if one match has less character errors than other, it is better.
    3. Else if one match has more correct matches, it is better.
    3. Otherwise, scores are the same.
    
    This class is used in the align() function that searches for the alignment with the best score.
    """
    n_word_errors: int = 0
    n_correct: int = 0
    n_char_errors: int = 0
    
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


@dataclass(kw_only=True, slots=True)
class Match:
    """
    An element of a string alignment: a match between zero or more words in two texts.
    - `true` array contains tokens from the first text
    - `pred` array contains tokens from the second text (both arrays cannot be empty simultaneously)
    - `status` is one of 'correct', 'deletion', 'insertion', 'replacement
    - `score` is the corresponding AlignmentScore
    
    Note that this class does not calculate or validate `status` or `score`, it only stores them.
    Use `match_from_pair()` to construct `Match` object and fill these fields.
    """
    true: list[Token]
    pred: list[Token]
    status: Literal['correct', 'deletion', 'insertion', 'replacement']
    score: AlignmentScore
    
    def __repr__(self) -> str:
        first = ' '.join([str(x) for x in self.true])
        second = ' '.join([str(x) for x in self.pred])
        return f'Match({first}, {second})'


@dataclass(slots=True)
class MatchesList:
    """
    An string alignment: a list of matches, such that:
    - A sum of `[m.true for m in self.matches]` give the full first text as a list of tokens
    - A sum of `[m.pred for m in self.matches]` give the full second text as a list of tokens
    
    If true text is multivariant, `[m.true for m in self.matches]` contains only a single
    variant for each multivariant block.
    
    `total_true_len` and `score` are filled automatically in the `MatchesList.from_list()`
    """
    matches: list[Match]
    total_true_len: int
    score: AlignmentScore
    
    @classmethod
    def from_list(cls, matches: list[Match]) -> MatchesList:
        return MatchesList(
            matches=matches,
            total_true_len=sum(_true_len(m) for m in matches),
            score=sum([m.score for m in matches], AlignmentScore())
        )
    
    def prepend(self, match: Match) -> MatchesList:
        return MatchesList(
            matches=[match] + self.matches,
            total_true_len=_true_len(match) + self.total_true_len,
            score = match.score + self.score
        )
    
    def append(self, match: Match) -> MatchesList:
        return MatchesList(
            matches=self.matches + [match],
            total_true_len=self.total_true_len + _true_len(match),
            score = self.score + match.score
        )
    
    def n_errors_with_insertions_tolerance(self, max_insertions: int = 4) -> int:
        n_errors = 0
        n_prev_insertions = 0
        for match in self.matches:
            if match.status != 'insertion':
                n_errors += n_prev_insertions
                n_prev_insertions = 0
                n_errors += match.score.n_word_errors
            else:
                n_prev_insertions = min(4, n_prev_insertions + len(match.pred))
        n_errors += n_prev_insertions
        return n_errors


def _true_len(match: Match) -> int:
    if len(match.true) == 1 and isinstance(match.true[0].value, Anything):
        return 0
    return len(match.true)