from __future__ import annotations

from functools import cache

import nltk

from .data import Anything, Token, MultiVariant, Match, MatchesList, AlignmentScore


@cache
def _char_edit_distance(true: str, pred: str) -> int:
    return nltk.edit_distance(true, pred) # type: ignore


def match_from_pair(true: list[Token], pred: list[Token]) -> Match:
    """
    Constructs `Match` object and fill `status` and `score` fields.
    
    This function does not solve an optimal alighment problem. If both word sequences are the same,
    or the first is `[Anything()]`, then match is considered 'correct', otherwise incorrect. In
    incorrect match, if both texts are not empty, it is considered 'replacement', otherwise
    'deletion' or 'insertion'.
    
    This is a helper function that `align()` uses to find an optimal alignment.
    """
    assert len(true) > 0 or len(pred) > 0
    is_anything = len(true) == 1 and isinstance(true[0].value, Anything)
    if [t.value for t in true] == [t.value for t in pred] or is_anything:
        status = 'correct'
    elif len(pred) == 0:
        status = 'deletion'
    elif len(true) == 0:
        status = 'insertion'
    else:
        status = 'replacement'
    return Match(
        true=true,
        pred=pred,
        status=status,
        score=AlignmentScore(
            n_word_errors=0 if status == 'correct' else max(len(true), len(pred)),
            n_correct=len(true) if status == 'correct' else 0,
            n_char_errors=_char_edit_distance(
                ' '.join(str(t.value) for t in true),
                ' '.join(str(t.value) for t in pred)
            ) if not is_anything else 0,
        ),
    )
    

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
    """
    Recursively solves an optimal alignment task.
    
    Evaluates alignments using AlignmentScore that first compares word errors, then character errors
    and number of corret matches. This means that across many alignments with minimal word error count,
    will select one with minimal character error count and maximum number of corret matches. This
    helps to improve alignments. This is especially important for streaming recognition, because to
    evaluate latency we need to obtain an alignment, not only WER or CER value.
    """
    assert all(isinstance(x, Token) for x in pred), 'prediction cannot be multivariant'
        
    multivariant_prefixes: dict[tuple[tuple[int, int], int], list[Token]] = {}
    for x in true:
        if isinstance(x, MultiVariant):
            for i, option in enumerate(x.options):
                multivariant_prefixes[x.pos, i] = option
    
    @cache
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
                    _matches.append(match_from_pair(shortest, []))
            return MatchesList.from_list(_matches)
        elif len(_pred) > 0 and len(_true) == 0:
            return MatchesList.from_list([
                match_from_pair([], [token])
                for token in _pred
            ])
        elif not isinstance(_true[0], MultiVariant):
            options: list[MatchesList] = []
            current_match_options = [
                # option 1: match true[0] with pred[0]
                (1, 1, match_from_pair([_true[0]], [_pred[0]])),
                # option 2: match pred[0] with nothing
                (0, 1, match_from_pair([], [_pred[0]])),
                # option 3: match true[0] with nothing
                (1, 0, match_from_pair([_true[0]], [])),
            ]
            for i, j, current_match in current_match_options:
                new_true_pos = true_pos
                new_multivariant_prefix_id = multivariant_prefix_id
                new_multivariant_prefix_pos = multivariant_prefix_pos
                if i == 1:
                    if multivariant_prefix_id is not None:
                        if len(prefix) > 1:  # pyright: ignore[reportPossiblyUnboundVariable]
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
                current_match = match_from_pair([_true[0]], [_pred[0]])
                options.append(
                    # option 4: match Anything with pred[0], but keep Anything in the true tokens
                    _align_recursive(
                        true_pos,
                        pred_pos + 1,
                        multivariant_prefix_id,
                        multivariant_prefix_pos,
                    ).prepend(current_match)
                )
            
            return max(options, key=lambda x: x.score)
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
            return max(options, key=lambda x: x.score)
    
    result = _align_recursive(0, 0, None, 0)
    # print(_align_recursive.cache_info()) # type: ignore
    return result