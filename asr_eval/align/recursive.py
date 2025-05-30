from __future__ import annotations
from functools import lru_cache

from .data import Anything, Token, MultiVariant, Match, MatchesList
    

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