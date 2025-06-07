from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import nltk # pyright: ignore[reportMissingTypeStubs]


@dataclass(slots=True)
class Anything:
    def __eq__(self, other: Any) -> bool:
        return True
    
    def __repr__(self) -> str:
        return '<*>'


@dataclass(slots=True)
class Token:
    value: str | Anything
    pos: tuple[int, int] = (0, 0)
    
    def __repr__(self) -> str:
        return f'Token({self.value}, {self.pos[0]}-{self.pos[1]})'


@dataclass(slots=True)
class MultiVariant:
    options: list[list[Token]]
    pos: tuple[int, int] = (0, 0)


@lru_cache(maxsize=None)
def _n_cerrs_inner(true: str, pred: str) -> int:
    return nltk.edit_distance(true, pred) # type: ignore


def _n_cerrs(true: list[Token], pred: list[Token]) -> int:
    if len(true) == 1 and isinstance(true[0].value, Anything):
        return 0
    return _n_cerrs_inner(''.join(str(t.value) for t in true), ''.join(str(t.value) for t in pred))


@dataclass(kw_only=True, slots=True)
class Match:
    true: list[Token]
    pred: list[Token]
    true_len: int
    n_errs: int
    n_cerrs: int
    
    @classmethod
    def from_pair(cls, true: list[Token], pred: list[Token]) -> Match:
        assert len(true) > 0 or len(pred) > 0
        match = Match(
            true=true,
            pred=pred,
            true_len=len(true),
            n_errs=0,
            n_cerrs=_n_cerrs(true, pred),
        )
        match.n_errs = 0 if match.get_status() == 'correct' else max(len(true), len(pred))
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
    total_n_errs: int
    total_n_correct: int
    total_n_cerrs: int
    
    @classmethod
    def from_list(cls, matches: list[Match]) -> MatchesList:
        return MatchesList(
            matches=matches,
            total_true_len=sum(m.true_len for m in matches),
            total_n_errs=sum(m.n_errs for m in matches),
            total_n_correct=sum(m.n_errs == 0 for m in matches),
            total_n_cerrs=sum(m.n_cerrs for m in matches),
        )
    
    def prepend(self, match: Match) -> MatchesList:
        return MatchesList(
            matches=[match] + self.matches,
            total_true_len=match.true_len + self.total_true_len,
            total_n_errs=match.n_errs + self.total_n_errs,
            total_n_correct=(match.n_errs == 0) + self.total_n_correct,
            total_n_cerrs=match.n_cerrs + self.total_n_cerrs,
        )
    
    @property
    def value(self) -> int:
        return self.total_n_errs * 100000000 + self.total_n_cerrs * 10000 - self.total_n_correct