from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .transcription import MultiVariantBlock, MultiVariantTranscription, SingleVariantTranscription, Token


class FlatLoc(Enum):
    Start = 0
    End = 1


@dataclass(slots=True)
class FlatView:
    positions: list[str | Literal[FlatLoc.Start, FlatLoc.End]]
    transitions: list[list[int]]
    resolved_multivariant_blocks: dict[tuple[int, int], list[tuple[str, int]]]


def flat_view(transcription: MultiVariantTranscription | SingleVariantTranscription) -> FlatView:
    '''
    A flat view, TODO use it in the `solve_optimal_alignment`. A flat view is
    - positions: [FlatLoc.Start] + list of token uids + [FlatLoc.End]
    - for each flat position except the last FlatLoc.End, list of allowed transitions
    - dict from transition (idx1, idx2) to a list of resolved multivariant blocks and options
    '''
    view = FlatView(
        positions=[FlatLoc.Start, FlatLoc.End],
        transitions=[[1]],
        resolved_multivariant_blocks=defaultdict(list),
    )
    
    # incrementally grow a flat view, initially empty
    for block in transcription.tokens:
        prev_end_pos = len(view.positions) - 1
        match block:
            case Token():
                view.positions = view.positions[:-1] + [block.uid, FlatLoc.End]
                view.transitions.append([prev_end_pos + 1])
            case MultiVariantBlock():
                view.positions = view.positions[:-1]  # cut FlatLoc.End
                paths_from_prefix = [i for i, to in enumerate(view.transitions) if prev_end_pos in to]
                options_end_positions: list[int] = []
                first_not_empty_option_added = False
                for option_idx, option in enumerate(block.options):
                    if len(option):
                        first_token_pos = len(view.positions)
                        # add paths from prefix to the current option
                        for i in paths_from_prefix:
                            if first_not_empty_option_added:
                                view.transitions[i].append(first_token_pos)
                            else:
                                # these paths were already added for the first option
                                assert first_token_pos in view.transitions[i]
                                first_not_empty_option_added = True
                            view.resolved_multivariant_blocks[(i, first_token_pos)].append(
                                (block.uid, option_idx)
                            )
                        # add option tokens and transitions between them
                        i = 0
                        for i, token in enumerate(option):
                            view.positions.append(token.uid)
                            view.transitions.append([])
                            if i > 0:
                                view.transitions[-2].append(first_token_pos + i)
                        # save option end position
                        options_end_positions.append(first_token_pos + i)
                # append FlatLoc.End
                view.positions.append(FlatLoc.End)
                assert len(view.positions) == len(view.transitions) + 1
                new_end_pos = len(view.positions) - 1
                # add paths from option endings to FlatLoc.End
                for option_end_pos in options_end_positions:
                    view.transitions[option_end_pos].append(new_end_pos)
                
                empty_options_indices = [i for i, option in enumerate(block.options) if len(option) == 0]
                if len(empty_options_indices) > 0:
                    # add paths from prefix to FlatLoc.End
                    for i in paths_from_prefix:
                        view.transitions[i].append(new_end_pos)
                        view.resolved_multivariant_blocks[(i, new_end_pos)] = [
                            # use [:-1] to remove a resolved index that was already added for this
                            # transition when processing non-empty options
                            *view.resolved_multivariant_blocks[(i, prev_end_pos)][:-1],
                            (block.uid, empty_options_indices[0])
                        ]
        
    assert all(len(lst) for lst in view.transitions)
    view.resolved_multivariant_blocks = dict(view.resolved_multivariant_blocks)
    return view


# TODO rewrite:
# 1) matrix with Pred and True axes
# 2) A custom mapping prev -> next for the True axis
# 3) store previous n errors (if known) and best prev cell in each cell
# 4) fill the matrix, then go backwards

# # value, uid
# TOKEN = tuple[str, str]

# MULTIVARIANT = list[list[TOKEN]]

# # true pos, pred pos, mv branch, mv pos
# POS_TYPE = tuple[int, int, int | None, int]

# # token uid, token uid, is error?, pos before match, pos after match
# MATCH_INFO = tuple[str | None, str | None, bool, POS_TYPE, POS_TYPE]

# # matches (if calculated), is solved?, matches leading to this pos
# POS_INFO = tuple[list[MATCH_INFO] | None, bool, list[MATCH_INFO] | None]


# def is_match(true_token: TOKEN, pred_token: TOKEN) -> bool:
#     return true_token[0] == '*' or true_token[0] == pred_token[0]


# def expand_matches(
#     true: list[TOKEN | MULTIVARIANT],
#     pred: list[TOKEN],
#     pos: POS_TYPE,
# ) -> list[MATCH_INFO]:
#     true_pos, pred_pos, mv_branch, mv_pos = pos
#     new_true_pos = true_pos
#     new_mv_branch = mv_branch
#     new_mv_pos = mv_pos
    
#     true_token: TOKEN | None = None
#     pred_token: TOKEN | None = None
#     is_anything = False
    
#     if mv_branch is not None:
#         mv = typing.cast(MULTIVARIANT, true[true_pos])
#         branch = mv[mv_branch]
#         true_token = branch[mv_pos]
#         if mv_pos < len(branch) - 1:
#             new_mv_pos += 1
#         else:
#             new_mv_branch = None
#             new_mv_pos = 0
#     else:
#         if true_pos < len(true):
#             true_token = typing.cast(TOKEN, true[true_pos])
#             if true_token[0] == '*':
#                 is_anything = True
#         new_true_pos += 1
    
#     if pred_pos < len(pred):
#         pred_token = pred[pred_pos]
    
#     options: list[MATCH_INFO] = []
    
#     if true_token is not None and pred_token is not None:
#         options.append((
#             true_token[1],
#             pred_token[1],
#             not is_match(true_token, pred_token),
#             pos,
#             (new_true_pos, pred_pos + 1, new_mv_branch, new_mv_pos),
#         ))
#         if is_anything:
#             options.append((
#                 true_token[1],
#                 pred_token[1],
#                 not is_match(true_token, pred_token),
#                 pos,
#                 (true_pos, pred_pos + 1, new_mv_branch, new_mv_pos),
#             ))
#     if true_token is not None:
#         options.append((
#             true_token[1],
#             None,
#             not is_anything,
#             pos,
#             (new_true_pos, pred_pos, new_mv_branch, new_mv_pos),
#         ))
#     if pred_token is not None:
#         options.append((
#             None,
#             pred_token[1],
#             True,
#             pos,
#             (true_pos, pred_pos + 1, new_mv_branch, new_mv_pos),
#         ))
#     return options

# def align(
#     true: list[TOKEN | MULTIVARIANT],
#     pred: list[TOKEN],
# ):
#     # TODO return something
#     positions: dict[POS_TYPE, POS_INFO] = {(0, 0, None, 0): (None, False, None)}
#     while True:
#         if all(matches is not None for _pos, (matches, _is_solved, _prev_matches) in positions.items()):
#             print('Cannot solve!')
#             return positions
#         for pos in list(positions):
#             matches, is_solved, prev_matches = positions[pos]
#             if matches is None:
#                 # expand matches
#                 matches = expand_matches(true, pred, pos)
#                 is_solved = all(
#                     after in positions and positions[after][1]
#                     for _, _, _, _, after in matches
#                 )
#                 positions[pos] = (matches, is_solved, prev_matches)
#                 # update other positions (add positions or references)
#                 for match in matches:
#                     _, _, _, _, pos_after = match
#                     if pos_after in positions:
#                         # add reference
#                         _, _, pos_before_prev_matches = positions[pos_after]
#                         typing.cast(list[MATCH_INFO], pos_before_prev_matches).append(match)
#                     else:
#                         # add new pos
#                         positions[pos_after] = (None, False, [match])
#                 # propagate is_solved back
#                 if is_solved:
#                     if prev_matches is None:
#                         print('Solved! (1)')
#                         return positions
#                     to_propagate = set(prev_matches)
#                     while len(to_propagate):
#                         match = to_propagate.pop()
#                         _, _, _, pos_before, _ = match
#                         pos_before_matches, _, pos_before_prev_matches = positions[pos_before]
#                         pos_before_matches = typing.cast(list[MATCH_INFO], pos_before_matches)
#                         pos_before_solved = all(positions[after][1] for _, _, _, _, after in pos_before_matches)
#                         if pos_before_solved:
#                             positions[pos_before] = pos_before_matches, True, pos_before_prev_matches
#                             if pos_before_prev_matches is None:
#                                 print('Solved! (2)')
#                                 return positions
#                             to_propagate.update(set(pos_before_prev_matches))