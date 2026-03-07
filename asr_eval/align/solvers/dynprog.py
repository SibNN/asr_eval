from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
import os
from typing import Literal, cast

import numpy as np
from tqdm.auto import tqdm

from asr_eval.align.matching import (
    AlignmentScore, Match, MatchesList, _match_from_pair, _true_len # pyright: ignore[reportPrivateUsage]
)
from asr_eval.align.transcription import (
    TOKEN_UID, Wildcard, MultiVariantBlock, Transcription, Token
)
from asr_eval.utils.table import Table2D
from asr_eval.utils.timer import Timer


__all__ = [
    'solve_optimal_alignment',
]


class _FlatLoc(Enum):
    Start = 0
    End = 1


@dataclass(slots=True)
class _FlatView:
    positions: list[str | Literal[_FlatLoc.Start, _FlatLoc.End]]
    transitions: list[list[int]]
    resolved_multivariant_blocks: dict[
        tuple[int, int],
        list[tuple[TOKEN_UID, int]],
    ]


@dataclass(slots=True)
class _CellData:
    score: AlignmentScore
    from_cell: tuple[int, int] | Literal['finish']
    from_match: Match | None


def solve_optimal_alignment(
    true: Transcription, pred: Transcription
) -> tuple[MatchesList, list[int]]:
    """
    Solves an optimal alignment task via dynamic programming. Uses a
    generalized version of the Needleman-Wunsch algorithm with the
    following modifications:
    
    1. Support for multivariant blocks in both texts.
    2. Support for :class:`~asr_eval.align.transcription.Wildcard`
       symbols in both texts.
    3. Better alignment due to the optimization of additional metrics
       (see the :class:`~asr_eval.align.matching.AlignmentScore` for
       details).
       
    The second returned value contains a selected option index for
    each multivariant block in `true`.

    Note:
        In the *asr_eval* workflow `pred` is single-variant. However,
        the algorithm supports multivariant blocks and
        :class:`~asr_eval.align.transcription.Wildcard` symbols for
        both `true` and `pred`.
        
    Example:
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> from asr_eval.align.solvers.dynprog import solve_optimal_alignment
        >>> true = 'hey <*> {eh} {one|1} {dollar|$}'
        >>> pred = 'Hey man eh dollar'
        >>> matches_list, selected_blocks = solve_optimal_alignment(
        ...     DEFAULT_PARSER.parse_transcription(true),
        ...     DEFAULT_PARSER.parse_transcription(pred),
        ... )
        >>> matches_list.score
        AlignmentScore(n_word_errors=1, n_correct=3, n_char_errors=1)
        >>> # selected #0 in {eh|}, #1 in {one|1}, #0 in {dollar|$}
        >>> selected_blocks
        [0, 1, 0]

    """
    profile = bool(os.environ.get('ASR_EVAL_SOLVER_PROFILE', False))
    with Timer(verbose='\tprepare' if profile else None):
        true_view = _flat_view(true)
        pred_view = _flat_view(pred)

        true_tokens = {token.uid: token for token in true.list_all_tokens()}
        pred_tokens = {token.uid: token for token in pred.list_all_tokens()}

        # the length of FlatView positions excluding FlatLoc.End
        true_view_len = len(true_view.transitions)
        pred_view_len = len(pred_view.transitions)

        # finish positions mask for the true and pred FlatView-s
        true_finish_mask = [true_view_len in t for t in true_view.transitions]
        pred_finish_mask = [pred_view_len in t for t in pred_view.transitions]

        assert sum(true_finish_mask)
        assert sum(pred_finish_mask)

    with Timer(verbose='\tmatches' if profile else None):
        # position (row, col) is a score for a match between
        # true_view.positions[row + 1] and pred_view.positions[col + 1]
        matches_for_replacements = Table2D[Match].construct_with_indices(
            rows=true_view_len - 1,
            cols=pred_view_len - 1,
            default=lambda row, col: _match_from_pair(
                true_tokens[cast(str, true_view.positions[row + 1])],
                pred_tokens[cast(str, pred_view.positions[col + 1])],
            ),
            pbar=profile,
        )

        # flags if tokens are Anything()
        # the first [False] is for FlatView.Start
        true_is_anything = [False] + [
            isinstance(
                true_tokens[cast(str, true_view.positions[row + 1])].value,
                Wildcard
            )
            for row in range(true_view_len - 1)
        ]
        # the first [False] is for FlatView.Start
        pred_is_anything = [False] + [
            isinstance(
                pred_tokens[cast(str, pred_view.positions[col + 1])].value,
                Wildcard
            )
            for col in range(pred_view_len - 1)
        ]

        # position (row,) is a score for a match between
        # true_view.positions[row + 1] and nothing
        matches_for_deletions = [
            _match_from_pair(
                true_tokens[cast(str, true_view.positions[row + 1])], None
            )
            for row in range(true_view_len - 1)
        ]

        # position (col,) is a score for a match between
        # nothing and pred_view.positions[col + 1]
        matches_for_insertions = [
            _match_from_pair(
                None, pred_tokens[cast(str, pred_view.positions[col + 1])]
            )
            for col in range(pred_view_len - 1)
        ]

    with Timer(verbose='\tmake table' if profile else None):
        # _CellData is an AlignmentScore for the tail, and a tuple of indices
        # from where we came here during a backward propagation; the table is
        # initially empty
        score_table = Table2D[_CellData | None].construct(
            rows=true_view_len, cols=pred_view_len, default=lambda: None
        )

        # initial values for score_table, where the tail is empty
        for row_idx in np.where(true_finish_mask)[0]:
            for col_idx in np.where(pred_finish_mask)[0]:
                score_table[row_idx, col_idx] = (
                    _CellData(AlignmentScore(), 'finish', None)
                )
    
    # backward propagation step that fills score_table[row_idx, col_idx]
    # can be called only for cells that are not filled yet
    def backprop_fill_score_table_cell(row_idx: int, col_idx: int):
        # determine sources for both axes
        true_source_indices = true_view.transitions[row_idx] + [row_idx]
        if true_finish_mask[row_idx]:
            true_source_indices.remove(true_view_len)
        pred_source_indices = pred_view.transitions[col_idx] + [col_idx]
        if pred_finish_mask[col_idx]:
            pred_source_indices.remove(pred_view_len)
        assert true_source_indices or pred_source_indices

        candidates: list[_CellData] = []

        for row_src_idx in true_source_indices:
            same_row = row_src_idx == row_idx
            for col_src_idx in pred_source_indices:
                same_col = col_src_idx == col_idx
                if same_row and same_col:
                    continue
                if same_row: # insertion (or match with <*> in true)
                    transition_match = (
                        matches_for_insertions[col_src_idx - 1]
                        if not true_is_anything[row_idx]
                        else matches_for_replacements[
                            row_idx - 1, col_src_idx - 1
                        ]
                    )
                elif same_col: # deletion (or match with <*> in pred)
                    transition_match = (
                        matches_for_deletions[row_src_idx - 1]
                        if not pred_is_anything[col_idx]
                        else matches_for_replacements[
                            row_src_idx - 1, col_idx - 1
                        ]
                    )
                else: # replacement or correct
                    transition_match = matches_for_replacements[
                        row_src_idx - 1, col_src_idx - 1
                    ]
                
                source_cell = score_table[row_src_idx, col_src_idx]
                assert isinstance(source_cell, _CellData)
                source_score = source_cell.score

                combined_score = source_score + transition_match.score
                candidates.append(_CellData(
                    combined_score,
                    (row_src_idx, col_src_idx),
                    transition_match,
                ))
        
        assert len(candidates)
        # max score is less errors
        best_candidate = max(candidates, key=lambda cell_data: cell_data.score)
        score_table[row_idx, col_idx] = best_candidate
    
    with Timer(verbose='\tfill table' if profile else None):
        # backward propagation with filling score_table
        for row_idx in tqdm(
            list(range(true_view_len))[::-1],
            disable=not profile,
        ):
            for col_idx in list(range(pred_view_len))[::-1]:
                if score_table[row_idx, col_idx] is None:
                    backprop_fill_score_table_cell(row_idx, col_idx)
    
    # now all cells are filled
    score_table = cast(Table2D[_CellData], score_table)

    with Timer(verbose='\tprocess results' if profile else None):
        # restoring the trajectory
        trajectory: list[tuple[int, int]] = [(0, 0)]
        while True:
            last_cell = score_table[trajectory[-1]]
            if last_cell.from_cell == 'finish':
                break
            trajectory.append(last_cell.from_cell)

        score = score_table[0, 0].score
        matches = [
            cast(Match, score_table[pos].from_match)
            for pos in trajectory[:-1]
        ]

        resolved_multivariant_blocks: dict[TOKEN_UID, int] = {}
        for (
            (row_idx1, _col_idx1), (row_idx2, _col_idx2)
        ) in pairwise(trajectory):
            resolved_multivariant_blocks |= dict(
                true_view.resolved_multivariant_blocks.get(
                    (row_idx1, row_idx2), []
                )
            )
        resolved_multivariant_indices = [
            # tail multivariant blocks may not exist in
            # `resolved_multivariant_blocks`, we select option 0 for them
            resolved_multivariant_blocks.get(block.uid, 0)
            for block in true.blocks
            if isinstance(block, MultiVariantBlock)
        ]

    return MatchesList(
        matches=matches,
        total_true_len=sum(_true_len(match) for match in matches),
        score=score,
    ), resolved_multivariant_indices


def _flat_view(transcription: Transcription) -> _FlatView:
    """
    A flat view:
    - positions: [FlatLoc.Start] + list of token uids + [FlatLoc.End]
    - for each flat position except the last FlatLoc.End, list of allowed transitions
    - dict from transition (idx1, idx2) to a list of resolved multivariant blocks and options
    """
    view = _FlatView(
        positions=[_FlatLoc.Start, _FlatLoc.End],
        transitions=[[1]],
        resolved_multivariant_blocks=defaultdict(list),
    )
    
    # incrementally grow a flat view, initially empty
    for block in transcription.blocks:
        prev_end_pos = len(view.positions) - 1
        match block:
            case Token():
                view.positions = (
                    view.positions[:-1] + [block.uid, _FlatLoc.End]
                )
                view.transitions.append([prev_end_pos + 1])
            case MultiVariantBlock():
                view.positions = view.positions[:-1]  # cut FlatLoc.End
                paths_from_prefix = [
                    i for i, to in enumerate(view.transitions)
                    if prev_end_pos in to
                ]
                # resolved multivariant blocks for each path from prefix
                resolved_from_prefix = [
                    view.resolved_multivariant_blocks[i, prev_end_pos]
                    for i in paths_from_prefix
                ]
                options_end_positions: list[int] = []
                first_not_empty_option_added = False
                for option_idx, option in enumerate(block.options):
                    if len(option):
                        first_token_pos = len(view.positions)
                        # add paths from prefix to the current option
                        for i, resolved in zip(
                            paths_from_prefix,
                            resolved_from_prefix,
                            strict=True,
                        ):
                            if first_not_empty_option_added:
                                view.transitions[i].append(first_token_pos)
                            else:
                                # these paths were already added for the
                                # first option
                                assert first_token_pos in view.transitions[i]
                                first_not_empty_option_added = True
                            view.resolved_multivariant_blocks[
                                i, first_token_pos
                            ] = resolved + [(block.uid, option_idx)]
                        # add option tokens and transitions between them
                        i = 0
                        for i, token in enumerate(option):
                            view.positions.append(token.uid)
                            view.transitions.append([])
                            if i > 0:
                                view.transitions[-2].append(
                                    first_token_pos + i
                                )
                        # save option end position
                        options_end_positions.append(first_token_pos + i)
                # append FlatLoc.End
                view.positions.append(_FlatLoc.End)
                assert len(view.positions) == len(view.transitions) + 1
                new_end_pos = len(view.positions) - 1
                # add paths from option endings to FlatLoc.End
                for option_end_pos in options_end_positions:
                    view.transitions[option_end_pos].append(new_end_pos)
                
                empty_options_indices = [
                    i for i, option in enumerate(block.options)
                    if len(option) == 0
                ]
                if len(empty_options_indices) > 0:
                    # add paths from prefix to FlatLoc.End
                    for i in paths_from_prefix:
                        view.transitions[i].append(new_end_pos)
                        view.resolved_multivariant_blocks[i, new_end_pos] = [
                            # use [:-1] to remove a resolved index that was
                            # already added for this transition when processing
                            # non-empty options
                            *view.resolved_multivariant_blocks \
                                [i, prev_end_pos][:-1],
                            (block.uid, empty_options_indices[0])
                        ]
        
    assert all(len(lst) for lst in view.transitions)
    view.resolved_multivariant_blocks = dict(view.resolved_multivariant_blocks)
    return view