from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal, cast

import nltk

from .transcription import Anything, Token, MultiVariantBlock


__all__ = [
    'match_from_pair',
    '_select_shortest_multi_variants',
    'solve_optimal_alignment',
    'AlignmentScore',
    'Match',
    'MatchesList',
]


@cache
def _char_edit_distance(true: str, pred: str) -> int:
    return nltk.edit_distance(true, pred) # type: ignore


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
    true: Token | None
    pred: Token | None
    status: Literal['correct', 'deletion', 'insertion', 'replacement']
    score: AlignmentScore

    def __repr__(self) -> str:
        first = str(self.true.value) if self.true is not None else ''
        second = str(self.pred.value) if self.pred is not None else ''
        return f'Match({first}, {second})'


def match_from_pair(true: Token | None, pred: Token | None) -> Match:
    """
    Constructs `Match` object and fill `status` and `score` fields.
    
    This function does not solve an optimal alighment problem. If both word sequences are the same,
    or the first is `[Anything()]`, then match is considered 'correct', otherwise incorrect. In
    incorrect match, if both texts are not empty, it is considered 'replacement', otherwise
    'deletion' or 'insertion'.
    
    This is a helper function that `align()` uses to find an optimal alignment.
    """
    T = true is not None
    P = pred is not None
    assert T or P
    
    is_anything = T and isinstance(true.value, Anything)
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
            n_char_errors=_char_edit_distance(
                str(true.value) if T else '',
                str(pred.value) if P else '',
            ) if (not is_anything and status != 'correct') else 0,
        ),
    )


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
    

def _select_shortest_multi_variants(seq: list[Token | MultiVariantBlock]) -> list[Token]:
    '''
    Selects the shortest option in each muultivariant block.
    '''
    result: list[Token] = []
    for x in seq:
        if isinstance(x, MultiVariantBlock):
            result += min(x.options, key=len)
        else:
            result.append(x)
    return result


def _true_len(match: Match) -> int:
    if match.true is not None and isinstance(match.true.value, Anything):
        # "Anything" blocks does not increment true len!
        return 0
    return int(match.true is not None)


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
                n_prev_insertions = min(max_insertions, n_prev_insertions + int(match.pred is not None))
        n_errors += n_prev_insertions
        return n_errors


def solve_optimal_alignment(
    true: list[Token | MultiVariantBlock] | list[Token],
    pred: list[Token],
    determine_selected_multivariant_indices: bool = True,
) -> tuple[MatchesList, list[int]]:
    """
    Recursively solves an optimal alignment task.
    
    Evaluates alignments using AlignmentScore that first compares word errors, then character errors
    and number of corret matches. This means that across many alignments with minimal word error count,
    will select one with minimal character error count and maximum number of corret matches. This
    helps to improve alignments. This is especially important for streaming recognition, because to
    evaluate latency we need to obtain an alignment, not only WER or CER value.
    
    Returns selected multivariant indices as the second returned value.
    
    If determine_selected_multivariant_indices=False, returns [] as the second returned value.
    This is useful for partial alignments where it is hard to determine which blocks were selected.
    TODO this shoud be fixed after rewriting the alignment algorithm to use flat view.
    """
    assert all(isinstance(x, Token) for x in pred), 'prediction cannot be multivariant'
        
    multivariant_prefixes: dict[tuple[str, int], list[Token]] = {}
    for x in true:
        if isinstance(x, MultiVariantBlock):
            for i, option in enumerate(x.options):
                multivariant_prefixes[x.uid, i] = option
    
    @cache
    def _align_recursive(
        true_pos: int,
        pred_pos: int,
        multivariant_prefix_id: tuple[str, int] | None,
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
                if len(shortest := _select_shortest_multi_variants([token])):
                    _matches += [match_from_pair(t, None) for t in shortest]
            return MatchesList.from_list(_matches)
        elif len(_pred) > 0 and len(_true) == 0:
            return MatchesList.from_list([
                match_from_pair(None, token)
                for token in _pred
            ])
        elif not isinstance(_true[0], MultiVariantBlock):
            options: list[MatchesList] = []
            current_match_options = [
                # option 1: match true[0] with pred[0]
                (1, 1, match_from_pair(_true[0], _pred[0])),
                # option 2: match pred[0] with nothing
                (0, 1, match_from_pair(None, _pred[0])),
                # option 3: match true[0] with nothing
                (1, 0, match_from_pair(_true[0], None)),
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
                current_match = match_from_pair(_true[0], _pred[0])
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
                    (_true[0].uid, i) if len(_true[0].options[i]) else None,
                    0,
                )
                for i in range(len(_true[0].options))
            ]
            return max(options, key=lambda x: x.score)
    
    result = _align_recursive(0, 0, None, 0)
    # print(_align_recursive.cache_info()) # type: ignore
    
    # determine which multivariant blocks were selected
    if determine_selected_multivariant_indices:
        pos = _TranscriptionPosition(0)
        selected_options: list[int] = []
        
        for match in result.matches:
            if match.true is not None:
                while True:
                    pos, selected_option_idx, selected_empty_option = (
                        pos.step_forward(true, match.true.uid)
                    )
                    if selected_option_idx is not None:
                        selected_options.append(selected_option_idx)
                    if not selected_empty_option:
                        break
        
        assert pos.in_multivariant_option is None
        
        while True:
            # step over the trailing empty multivariant blocks
            try:
                pos, selected_option_idx, selected_empty_option = pos.step_forward(true, '')
                assert selected_empty_option
                assert selected_option_idx is not None
                selected_options.append(selected_option_idx)
            except StopIteration:
                break
    else:
        selected_options = []
    
    return result, selected_options


@dataclass(slots=True)
class _TranscriptionPosition:
    '''
    A position between two tokens in a multivariant or single-variant transcription.
    `before_option_index` should be filled only if `in_multivariant_option=True`
    
    `in_multivariant_option` should be True only if we are **between** two tokens in a
    multivariant option. `before_option_index` should be filled accordingly, and.
    `current_pos` should point to the current multivariant block index. Otherwise,
    `before_option_index` should be None.
    
    This class allows to traverse a multivariant transcription.
    
    TODO use a flat view instead, here and in the `solve_optimal_alignment`
    '''
    before_index: int
    in_multivariant_option: int | None = None
    before_option_index: int | None = None

    def step_forward(
        self,
        tokens: list[Token | MultiVariantBlock] | list[Token],
        next_token_uid: str
    ) -> tuple[_TranscriptionPosition, int | None, bool]:
        '''
        Given a _TranscriptionPosition between two tokens, and the expected next token,
        does a step forward.
        
        If we enter a multivariant block, selects which option to choose and returns the index
        of this option as the second returned value.
        
        Otherwise just asserts that `next_token_uid` is the same as expected and returns None
        as the second returned value.
        
        The third returned value indicates that we did a step over an empty option and did not
        consume a token.
        
        If pos is after the last token in the `tokens`, raises StopIteration
        '''
        if self.in_multivariant_option is not None:
            # between two tokens in a multivariant option
            assert self.before_option_index is not None
            block = tokens[self.before_index]
            assert isinstance(block, MultiVariantBlock)
            option = block.options[self.in_multivariant_option]
            next_token = option[self.before_option_index]
            assert next_token.uid == next_token_uid
            if self.before_option_index == len(option) - 1:
                # we step over the last token in the current option
                # reset in_multivariant_option to None
                return _TranscriptionPosition(self.before_index + 1, None, None), None, False
            else:
                return _TranscriptionPosition(
                    self.before_index, self.in_multivariant_option, self.before_option_index + 1
                ), None, False
        else:
            # not inside a multivariant block (possibly before or after it)
            assert self.before_option_index is None
            if self.before_index >= len(tokens):
                raise StopIteration
            block = tokens[self.before_index]
            if isinstance(block, Token):
                # step over a token
                assert block.uid == next_token_uid
                return _TranscriptionPosition(self.before_index + 1, None, None), None, False
            else:
                # enter or step over a multivariant block
                empty_option_idx: int | None = None
                for option_idx, option in enumerate(block.options):
                    if len(option) == 0:
                        empty_option_idx = option_idx
                    elif len(option) and option[0].uid == next_token_uid:
                        # enter the current option
                        if len(option) == 1:
                            # step over the block
                            return _TranscriptionPosition(self.before_index + 1, None, None), option_idx, False
                        else:
                            # step inside the block
                            return _TranscriptionPosition(self.before_index, option_idx, 1), option_idx, False
                else:
                    # did not find an option
                    # check that we have an empty option
                    assert empty_option_idx is not None
                    # step over the empty option
                    return _TranscriptionPosition(self.before_index + 1, None, None), empty_option_idx, True