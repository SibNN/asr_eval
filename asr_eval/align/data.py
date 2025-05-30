from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


@dataclass(kw_only=True, slots=True)
class Match:
    true: list[Token]
    pred: list[Token]
    true_len: int
    n_errs: int
    
    @classmethod
    def from_pair(cls, true: list[Token], pred: list[Token]) -> Match:
        assert len(true) > 0 or len(pred) > 0
        return Match(
            true=true,
            pred=pred,
            true_len=len(true),
            n_errs=(
                0
                if [t.value for t in true] == [t.value for t in pred]
                or (len(true) == 1 and isinstance(true[0].value, Anything))
                else max(len(true), len(pred))
            ),
        )
    
    def __repr__(self) -> str:
        first = ' '.join([str(x) for x in self.true])
        second = ' '.join([str(x) for x in self.pred])
        return f'({first}, {second})'


@dataclass(slots=True)
class MatchesList:
    matches: list[Match]
    total_true_len: int
    total_n_errs: int
    total_n_correct: int
    
    @classmethod
    def from_list(cls, matches: list[Match]) -> MatchesList:
        return MatchesList(
            matches=matches,
            total_true_len=sum(m.true_len for m in matches),
            total_n_errs=sum(m.n_errs for m in matches),
            total_n_correct=sum(m.n_errs == 0 for m in matches),
        )
    
    def prepend(self, match: Match) -> MatchesList:
        return MatchesList(
            matches=[match] + self.matches,
            total_true_len=match.true_len + self.total_true_len,
            total_n_errs=match.n_errs + self.total_n_errs,
            total_n_correct=(match.n_errs == 0) + self.total_n_correct,
        )
    
    @property
    def value(self) -> int:
        return self.total_n_errs * 1_000_000 - self.total_n_correct