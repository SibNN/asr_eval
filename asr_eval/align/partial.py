from .data import MatchesList, MultiVariant, Token
from .recursive import align


__all__ = [
    "align_partial",
    "get_starting_part",
]


def align_partial(
    true: list[Token | MultiVariant], pred: list[Token], time: float
) -> MatchesList:
    '''
    Aligns a starting part of `true` up to `time` agains `pred`. Requies
    `true` to be timed.
    
    See details in the `get_starting_part` docstring.
    '''
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