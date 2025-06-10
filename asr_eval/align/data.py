from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, cast

import nltk
import numpy as np

from ..utils import new_uid


@dataclass(slots=True)
class Anything:
    def __eq__(self, other: Any) -> bool:
        return True
    
    def __repr__(self) -> str:
        return '<*>'


@dataclass(slots=True)
class Token:
    value: str | Anything
    uid: str = field(default_factory=new_uid)
    start_pos: int = 0
    end_pos: int = 0
    start_time: float = np.nan
    end_time: float = np.nan
    
    def __repr__(self) -> str:
        # return f'Token({self.value}, {self.pos[0]}-{self.pos[1]})'
        return f'Token({self.value})'


@dataclass(slots=True)
class MultiVariant:
    options: list[list[Token]]
    pos: tuple[int, int] = (0, 0)


@dataclass(slots=True)
class AlignmentScore:
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


@lru_cache(maxsize=None)
def _edit_distance(true: str, pred: str) -> int:
    return nltk.edit_distance(true, pred) # type: ignore


def _n_cerrs(true: list[Token], pred: list[Token]) -> int:
    if len(true) == 1 and isinstance(true[0].value, Anything):
        return 0
    return _edit_distance(' '.join(str(t.value) for t in true), ' '.join(str(t.value) for t in pred))


@dataclass(kw_only=True, slots=True)
class Match:
    true: list[Token]
    pred: list[Token]
    true_len: int
    score: AlignmentScore
    
    @classmethod
    def from_pair(cls, true: list[Token], pred: list[Token]) -> Match:
        assert len(true) > 0 or len(pred) > 0
        match = Match(
            true=true,
            pred=pred,
            true_len=len(true),
            score=AlignmentScore(),
        )
        is_correct = match.get_status() == 'correct'
        match.score.n_word_errors = 0 if is_correct else max(len(true), len(pred))
        match.score.n_correct = match.score.n_word_errors == 0
        match.score.n_char_errors = _n_cerrs(true, pred)
        return match
    
    def __repr__(self) -> str:
        first = ' '.join([str(x) for x in self.true])
        second = ' '.join([str(x) for x in self.pred])
        return f'({first}, {second})'
    
    def get_status(self) -> Literal['correct', 'deletion', 'insertion', 'replacement']:
        if (
            [t.value for t in self.true] == [t.value for t in self.pred]
            or (len(self.true) == 1 and isinstance(self.true[0].value, Anything))
        ):
            return 'correct'
        elif len(self.pred) == 0:
            return 'deletion'
        elif len(self.true) == 0:
            return 'insertion'
        else:
            return 'replacement'


@dataclass(slots=True)
class MatchesList:
    matches: list[Match]
    total_true_len: int
    score: AlignmentScore
    
    @classmethod
    def from_list(cls, matches: list[Match]) -> MatchesList:
        return MatchesList(
            matches=matches,
            total_true_len=sum(m.true_len for m in matches),
            score=sum([m.score for m in matches], AlignmentScore())
        )
    
    def prepend(self, match: Match) -> MatchesList:
        return MatchesList(
            matches=[match] + self.matches,
            total_true_len=match.true_len + self.total_true_len,
            score = match.score + self.score
        )