import typing


# TODO return something from align(), write tests

# TODO rewrite:
# 1) matrix with Pred and True axes
# 2) A custom mapping prev -> next for the True axis
# 3) store previous n errors (if known) and best prev cell in each cell
# 4) fill the matrix, then go backwards

# value, uid
TOKEN = tuple[str, str]

MULTIVARIANT = list[list[TOKEN]]

# true pos, pred pos, mv branch, mv pos
POS_TYPE = tuple[int, int, int | None, int]

# token uid, token uid, is error?, pos before match, pos after match
MATCH_INFO = tuple[str | None, str | None, bool, POS_TYPE, POS_TYPE]

# matches (if calculated), is solved?, matches leading to this pos
POS_INFO = tuple[list[MATCH_INFO] | None, bool, list[MATCH_INFO] | None]


def is_match(true_token: TOKEN, pred_token: TOKEN) -> bool:
    return true_token[0] == '*' or true_token[0] == pred_token[0]


def expand_matches(
    true: list[TOKEN | MULTIVARIANT],
    pred: list[TOKEN],
    pos: POS_TYPE,
) -> list[MATCH_INFO]:
    true_pos, pred_pos, mv_branch, mv_pos = pos
    new_true_pos = true_pos
    new_mv_branch = mv_branch
    new_mv_pos = mv_pos
    
    true_token: TOKEN | None = None
    pred_token: TOKEN | None = None
    is_anything = False
    
    if mv_branch is not None:
        mv = typing.cast(MULTIVARIANT, true[true_pos])
        branch = mv[mv_branch]
        true_token = branch[mv_pos]
        if mv_pos < len(branch) - 1:
            new_mv_pos += 1
        else:
            new_mv_branch = None
            new_mv_pos = 0
    else:
        if true_pos < len(true):
            true_token = typing.cast(TOKEN, true[true_pos])
            if true_token[0] == '*':
                is_anything = True
        new_true_pos += 1
    
    if pred_pos < len(pred):
        pred_token = pred[pred_pos]
    
    options: list[MATCH_INFO] = []
    
    if true_token is not None and pred_token is not None:
        options.append((
            true_token[1],
            pred_token[1],
            not is_match(true_token, pred_token),
            pos,
            (new_true_pos, pred_pos + 1, new_mv_branch, new_mv_pos),
        ))
        if is_anything:
            options.append((
                true_token[1],
                pred_token[1],
                not is_match(true_token, pred_token),
                pos,
                (true_pos, pred_pos + 1, new_mv_branch, new_mv_pos),
            ))
    if true_token is not None:
        options.append((
            true_token[1],
            None,
            not is_anything,
            pos,
            (new_true_pos, pred_pos, new_mv_branch, new_mv_pos),
        ))
    if pred_token is not None:
        options.append((
            None,
            pred_token[1],
            True,
            pos,
            (true_pos, pred_pos + 1, new_mv_branch, new_mv_pos),
        ))
    return options

def align(
    true: list[TOKEN | MULTIVARIANT],
    pred: list[TOKEN],
):
    # TODO return something
    positions: dict[POS_TYPE, POS_INFO] = {(0, 0, None, 0): (None, False, None)}
    while True:
        if all(matches is not None for _pos, (matches, _is_solved, _prev_matches) in positions.items()):
            print('Cannot solve!')
            return positions
        for pos in list(positions):
            matches, is_solved, prev_matches = positions[pos]
            if matches is None:
                # expand matches
                matches = expand_matches(true, pred, pos)
                is_solved = all(
                    after in positions and positions[after][1]
                    for _, _, _, _, after in matches
                )
                positions[pos] = (matches, is_solved, prev_matches)
                # update other positions (add positions or references)
                for match in matches:
                    _, _, _, _, pos_after = match
                    if pos_after in positions:
                        # add reference
                        _, _, pos_before_prev_matches = positions[pos_after]
                        typing.cast(list[MATCH_INFO], pos_before_prev_matches).append(match)
                    else:
                        # add new pos
                        positions[pos_after] = (None, False, [match])
                # propagate is_solved back
                if is_solved:
                    if prev_matches is None:
                        print('Solved! (1)')
                        return positions
                    to_propagate = set(prev_matches)
                    while len(to_propagate):
                        match = to_propagate.pop()
                        _, _, _, pos_before, _ = match
                        pos_before_matches, _, pos_before_prev_matches = positions[pos_before]
                        pos_before_matches = typing.cast(list[MATCH_INFO], pos_before_matches)
                        pos_before_solved = all(positions[after][1] for _, _, _, _, after in pos_before_matches)
                        if pos_before_solved:
                            positions[pos_before] = pos_before_matches, True, pos_before_prev_matches
                            if pos_before_prev_matches is None:
                                print('Solved! (2)')
                                return positions
                            to_propagate.update(set(pos_before_prev_matches))