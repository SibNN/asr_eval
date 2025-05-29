from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import lru_cache
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


def parse_string(
    text: str,
    pos_shift: int = 0,
) -> list[Token]:
    result: list[Token] = []
    for match in re.finditer(r'\S+', text):
        word = match.group()
        start, end = match.span()
        result.append(Token(
            value=word if word != '<*>' else Anything(),
            pos=(start + pos_shift, end + pos_shift),
        ))
    return result

MULTIVARIANT_PATTERN = re.compile(
    r'({[^{}]*?})'            # multi variant
    '|'
    r'(?<=})([^{}]+?)(?={)'   # single variant
)


def parse_multi_variant_string(
    text: str,
) -> list[Token | MultiVariant]:
    result: list[Token | MultiVariant] = []
    for match in re.finditer(MULTIVARIANT_PATTERN, '}' + text + '{'):
        text_part = match.group()
        start = match.start() - 1  # account for '}' (see in re.finditer)
        end = match.end() - 1      # account for '}' (see in re.finditer)
        if text_part.startswith('{'):
            if start > 0:
                assert (c := text[start - 1]) in string.whitespace, (
                    f'put a space before a multivariant block, got "{c}"'
                )
            if end < len(text):
                assert (c := text[end]) in string.whitespace, (
                    f'put a space after a multivariant block, got "{c}"'
                )
            result.append(MultiVariant(
                options=[
                    parse_string(option.group(1), pos_shift=start + option.start() + 1)
                    for option in re.finditer(r'([^\|]*)\|', text_part[1:-1] + '|')
                ],
                pos=(start, end),
            ))
        else:
            result += parse_string(text_part, pos_shift=start)
    return result


@dataclass(kw_only=True)
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


@dataclass
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
    

def select_shortest_multi_variants(seq: list[Token | MultiVariant]) -> list[Token]:
    result: list[Token] = []
    for x in seq:
        if isinstance(x, MultiVariant):
            result += min(x.options, key=len)
        else:
            result.append(x)
    return result


def align(
    true: list[Token | MultiVariant],
    pred: list[Token],
) -> MatchesList:
    assert all(isinstance(x, Token) for x in pred), 'prediction cannot be multivariant'
        
    multivariant_prefixes: dict[tuple[tuple[int, int], int], list[Token]] = {}
    for x in true:
        if isinstance(x, MultiVariant):
            for i, option in enumerate(x.options):
                multivariant_prefixes[x.pos, i] = option
    
    @lru_cache(maxsize=None)
    def _align_recursive(
        true_pos: int,
        pred_pos: int,
        multivariant_prefix_id: tuple[tuple[int, int], int] | None,
        multivariant_prefix_pos: int,
    ) -> MatchesList:
        _true = true[true_pos:]
        _pred = pred[pred_pos:]
        
        if multivariant_prefix_id is not None:
            prefix = multivariant_prefixes[multivariant_prefix_id][multivariant_prefix_pos:]
            _true = prefix + _true
        
        if len(_pred) == 0 and len(_true) == 0:
            return MatchesList.from_list([])
        elif len(_pred) == 0 and len(_true) > 0:
            _matches: list[Match] = []
            for token in _true:
                if len(shortest := select_shortest_multi_variants([token])):
                    _matches.append(Match.from_pair(shortest, []))
            return MatchesList.from_list(_matches)
        elif len(_pred) > 0 and len(_true) == 0:
            return MatchesList.from_list([
                Match.from_pair([], [token])
                for token in _pred
            ])
        elif not isinstance(_true[0], MultiVariant):
            options: list[MatchesList] = []
            current_match_options = [
                # option 1: match true[0] with pred[0]
                (1, 1, Match.from_pair(_true[:1], _pred[:1])), # type: ignore
                # option 2: match pred[0] with nothing
                (0, 1, Match.from_pair([], _pred[:1])),
                # option 3: match true[0] with nothing
                (1, 0, Match.from_pair(_true[:1], [])), # type: ignore
            ]
            for i, j, current_match in current_match_options:
                new_true_pos = true_pos
                new_multivariant_prefix_id = multivariant_prefix_id
                new_multivariant_prefix_pos = multivariant_prefix_pos
                if i == 1:
                    if multivariant_prefix_id is not None:
                        if len(prefix) > 1: # type: ignore
                            new_multivariant_prefix_pos += 1
                        else:
                            new_multivariant_prefix_id = None
                            new_multivariant_prefix_pos = 0
                    else:
                        new_true_pos += 1
                _results = _align_recursive(
                        new_true_pos,
                        pred_pos + j,
                        new_multivariant_prefix_id,
                        new_multivariant_prefix_pos,
                    )
                options.append(
                    _results.prepend(current_match)
                )
            if isinstance(_true[0].value, Anything):
                current_match = Match.from_pair(_true[:1], _pred[:1]) # type: ignore
                options.append(
                    # option 4: match Anything with pred[0], but keep Anything in the true tokens
                    _align_recursive(
                        true_pos,
                        pred_pos + 1,
                        multivariant_prefix_id,
                        multivariant_prefix_pos,
                    ).prepend(current_match)
                )
            
            return min(options, key=lambda x: x.value)
        else:
            assert multivariant_prefix_id is None
            options = [
                _align_recursive(
                    true_pos + 1,
                    pred_pos,
                    (_true[0].pos, i) if len(_true[0].options[i]) else None,
                    0,
                )
                for i in range(len(_true[0].options))
            ]
            return min(options, key=lambda x: x.value)
    
    result = _align_recursive(0, 0, None, 0)
    # print(_align_recursive.cache_info()) # type: ignore
    return result