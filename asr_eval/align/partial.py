from typing import Sequence, cast

import numpy as np

from asr_eval.align.data import MatchesList, MultiVariant, Token
from asr_eval.align.recursive import align


def words_count(
    word_timings: Sequence[Token],
    time: float,
) -> tuple[int, bool]:
    '''
    Given a list of Token with `.start_time` and `.end_time` filled, returns a tuple of:
    1. Number of full words in the time span [0, time]
    2. `in_word` flag: is the given time inside a word?
    '''
    count = 0
    in_word = False

    for token in word_timings:
        if token.end_time <= time:
            count += 1
        else:
            if token.start_time < time:
                in_word = True
            break

    return count, in_word


def align_partial(
    true: list[Token],
    pred: list[Token],
    seconds_processed: float,
) -> MatchesList:
    """
    Aligns `pred` with the beginning part of `true` up to `seconds_processed`.

    In `true`, timings for all tokens should be filled (.start_time and .end_time).
    """
    for token in true:
        assert not np.isnan(token.start_time) and not np.isnan(token.end_time)

    n_true_words, in_true_word = words_count(true, seconds_processed)

    option1 = true[:n_true_words]
    alignment = align(cast(list[Token | MultiVariant], option1), pred)

    if in_true_word:
        option2 = true[:n_true_words + 1]
        alignment2 = align(cast(list[Token | MultiVariant], option2), pred)

        if alignment2.score > alignment.score:
            alignment = alignment2

    return alignment