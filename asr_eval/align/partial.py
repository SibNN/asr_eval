from .data import MatchesList, MultiVariant, Token
from .recursive import align


def align_partial(
    true: list[Token | MultiVariant], pred: list[Token], time: float
) -> MatchesList:
    return align(get_starting_part(true, time), pred)


def get_starting_part(
    tokens: list[Token | MultiVariant], time: float
) -> list[Token | MultiVariant]:
    '''
    Cut a multivariant string up to the specified time.
    
    If `time` is inside some token, then converts it into the multivariant
    block with options [token] and [].
    
    For example, let tokens = [A, B], token A spans from 1.0 to 2.0 and B
    spans from 3.0 to 4.0. Then `get_starting_part(tokens, time=3.5)` returns
    [A, MultiVariant(X)], where X = [[B], []].
    
    If `time` is inside an existing multivariant block, then cut each option
    up to the `time`, and if `time` is inside some token in some option, add
    another option with this token excluded.
    
    For example, let tokens = [A, MultiVariant([[B1], [B2, B3]])], and B1 spans
    from 3.0 to 4.0, B2 spans from 3.0 to 3.5, B3 spans from 3.5 to 4.0. Then
    `get_starting_part(time=3.7)` returns [A, MultiVariant(X)], where
    X = [[], [B1], [B2], [B2, B3]]. Here [] was obtained from cutting option
    [B1] and [B2] was obtained from cutting option [B2, B3].
    '''
    tokens_partial_stem: list[Token | MultiVariant] = []
    tokens_partial_tail_options: list[list[Token]] = []

    for block in tokens:
        if block.start_time >= time:
            break
        elif block.end_time <= time:
            tokens_partial_stem.append(block)
        else:
            # `time` is inside the block
            if isinstance(block, Token):
                tokens_partial_tail_options.append([block])
                tokens_partial_tail_options.append([])
            else:
                for option in block.options:
                    option_partial = [t for t in option if t.start_time < time]
                    tokens_partial_tail_options.append(option_partial)
                    if len(option_partial) and option_partial[-1].end_time > time:
                        tokens_partial_tail_options.append(option_partial[:-1])
            break

    if len(tokens_partial_tail_options):
        return tokens_partial_stem + [MultiVariant(tokens_partial_tail_options)]
    else:
        return tokens_partial_stem


# def get_tail_part(
#     tokens: list[Token | MultiVariant], after_token_uid: str
# ) -> list[Token | MultiVariant]:
#     '''
#     Returns a tail part of `tokens` starting from `after_token_uid` not inclusive.
#     If `after_token_uid` is a part of a multivariant block, selects its option
#     and prepends the remaining tokens (if any) to the result.
#     '''
#     result: list[Token | MultiVariant] = []
#     for block in tokens[::-1]:
#         match block:
#             case Token():
#                 if block.uid == after_token_uid:
#                     break
#             case MultiVariant():
#                 found_option: list[Token] | None = None
#                 for option in block.options:
#                     if any(t.uid == after_token_uid for t in option):
#                         found_option = option
#                         break
#                 if found_option is not None:
#                     while found_option[0].uid != after_token_uid:
#                         found_option = found_option[1:]
#                     result = found_option[1:] + result
#                     break
                
#         result.insert(0, block)
    
#     return result


# def words_count(
#     word_timings: Sequence[Token],
#     time: float,
# ) -> tuple[int, bool]:
#     '''
#     Given a list of Token with `.start_time` and `.end_time` filled, returns a tuple of:
#     1. Number of full words in the time span [0, time]
#     2. `in_word` flag: is the given time inside a word?
#     '''
#     count = 0
#     in_word = False

#     for token in word_timings:
#         if token.end_time <= time:
#             count += 1
#         else:
#             if token.start_time < time:
#                 in_word = True
#             break

#     return count, in_word


# def align_partial(
#     true: list[Token],
#     pred: list[Token],
#     seconds_processed: float,
# ) -> MatchesList:
#     """
#     Aligns `pred` with the beginning part of `true` up to `seconds_processed`.

#     In `true`, timings for all tokens should be filled (.start_time and .end_time).
    
#     TODO adapt for multivariant case.
#     """
#     for token in true:
#         assert token.is_timed

#     n_true_words, in_true_word = words_count(true, seconds_processed)
    
#     true_partial = cast(list[Token | MultiVariant], true[:n_true_words])
#     if in_true_word:
#         last_word = true[n_true_words]
#         true_partial.append(MultiVariant(
#             options=[[last_word], []],
#             pos=(last_word.start_pos, last_word.end_pos),
#         ))
    
#     return align(true_partial, pred)