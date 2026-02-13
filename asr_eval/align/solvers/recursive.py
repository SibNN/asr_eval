from __future__ import annotations
from dataclasses import dataclass
from functools import cache

from asr_eval.align.matching import (
    Match, MatchesList, _match_from_pair, _select_shortest_multi_variants # pyright: ignore[reportPrivateUsage]
)
from asr_eval.align.transcription import (
    TOKEN_UID,
    Wildcard,
    MultiVariantBlock,
    Transcription,
    SingleVariantTranscription,
    Token,
)


__all__ = [
    'solve_optimal_alignment',
]


def solve_optimal_alignment(
    true: Transcription,
    pred: SingleVariantTranscription,
    determine_selected_multivariant_indices: bool = True,
) -> tuple[MatchesList, list[int]]:
    """Solves an optimal alignment task via recursion. Supports
    multivariant annotations with
    :class:`~asr_eval.align.transcription.Wildcard` insertions.
    
    Note:
        This method is legacy, consired using
        :func:`asr_eval.align.solvers.dynprog.solve_optimal_alignment`
        instead.
    
    The last returned value is a selected option index for each
    multivariant block, if present in the ground truth.
    """
    true_blocks = list(true.blocks)
    pred_blocks = list(pred.blocks)

    assert all(isinstance(x, Token) for x in pred_blocks), (
        'prediction cannot be multivariant'
    )

    multivariant_prefixes: dict[tuple[TOKEN_UID, int], list[Token]] = {}
    for x in true_blocks:
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
        _true = true_blocks[true_pos:]
        _pred = pred_blocks[pred_pos:]

        if multivariant_prefix_id is not None:
            prefix = multivariant_prefixes \
                [multivariant_prefix_id][multivariant_prefix_pos:]
            _true = prefix + _true

        if len(_pred) == 0 and len(_true) == 0:
            return MatchesList.from_list([])
        elif len(_pred) == 0 and len(_true) > 0:
            _matches: list[Match] = []
            for token in _true:
                if len(shortest := _select_shortest_multi_variants([token])):
                    _matches += [_match_from_pair(t, None) for t in shortest]
            return MatchesList.from_list(_matches)
        elif len(_pred) > 0 and len(_true) == 0:
            return MatchesList.from_list([
                _match_from_pair(None, token)
                for token in _pred
            ])
        elif not isinstance(_true[0], MultiVariantBlock):
            options: list[MatchesList] = []
            current_match_options = [
                # option 1: match true[0] with pred[0]
                (1, 1, _match_from_pair(_true[0], _pred[0])),
                # option 2: match pred[0] with nothing
                (0, 1, _match_from_pair(None, _pred[0])),
                # option 3: match true[0] with nothing
                (1, 0, _match_from_pair(_true[0], None)),
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
            if isinstance(_true[0].value, Wildcard):
                current_match = _match_from_pair(_true[0], _pred[0])
                options.append(
                    # option 4: match Anything with pred[0], but keep Anything
                    # in the true tokens
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

        for match_idx, match in enumerate(result.matches):
            if match.true is not None:
                if (
                    match_idx > 0
                    and match.true is result.matches[match_idx - 1].true
                ):
                    # same true token for consecutive matches, happens with
                    # Token(Wilcard())
                    continue
                while True:
                    pos, selected_option_idx, selected_empty_option = (
                        pos.step_forward(true_blocks, match.true.uid)
                    )
                    if selected_option_idx is not None:
                        selected_options.append(selected_option_idx)
                    if not selected_empty_option:
                        break

        assert pos.in_multivariant_option is None

        while True:
            # step over the trailing empty multivariant blocks
            try:
                pos, selected_option_idx, selected_empty_option = (
                    pos.step_forward(true_blocks, '')
                )
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
    """
    A position between two tokens in a multivariant or single-variant
    transcription. `before_option_index` should be filled only if
    `in_multivariant_option=True`

    `in_multivariant_option` should be True only if we are **between**
    two tokens in a multivariant option. `before_option_index` should be
    filled accordingly, and. `current_pos` should point to the current
    multivariant block index. Otherwise, `before_option_index` should be
    None.

    This class allows to traverse a multivariant transcription.

    TODO use a flat view instead, here and in the
    `solve_optimal_alignment`
    """
    before_index: int
    in_multivariant_option: int | None = None
    before_option_index: int | None = None

    def step_forward(
        self,
        tokens: list[Token | MultiVariantBlock] | list[Token],
        next_token_uid: str
    ) -> tuple[_TranscriptionPosition, int | None, bool]:
        """
        Given a _TranscriptionPosition between two tokens, and the
        expected next token, does a step forward.

        If we enter a multivariant block, selects which option to choose
        and returns the index of this option as the second returned
        value.

        Otherwise just asserts that `next_token_uid` is the same as
        expected and returns None as the second returned value.

        The third returned value indicates that we did a step over an
        empty option and did not consume a token.

        If pos is after the last token in the `tokens`, raises
        StopIteration.
        """
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
                return (
                    _TranscriptionPosition(self.before_index + 1, None, None),
                    None,
                    False,
                )
            else:
                return _TranscriptionPosition(
                    self.before_index,
                    self.in_multivariant_option,
                    self.before_option_index + 1,
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
                return _TranscriptionPosition(self.before_index + 1, None, \
                    None), None, False
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
                            return _TranscriptionPosition(self.before_index \
                                + 1, None, None), option_idx, False
                        else:
                            # step inside the block
                            return _TranscriptionPosition(self.before_index, \
                                option_idx, 1), option_idx, False
                else:
                    # did not find an option
                    # check that we have an empty option
                    assert empty_option_idx is not None
                    # step over the empty option
                    return _TranscriptionPosition(self.before_index + 1, \
                        None, None), empty_option_idx, True