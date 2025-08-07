from itertools import pairwise

import matplotlib.pyplot as plt

from .data import Anything, MultiVariant, Token
from ..utils.plots import draw_bezier, draw_line_with_ticks


__all__ = [
    'draw_timed_transcription',
]


def draw_timed_transcription(
    transcription: list[Token | MultiVariant],
    y_pos: float = 0,
    y_delta: float = -1,
    y_tick_width: float = 0.1,
    ax: plt.Axes | None = None,
    graybox_y: tuple[float, float] | None = None,
):
    '''
    Draws a timed multivariant or single-variant transcription
    '''
    ax = ax or plt.gca()

    if len(transcription) == 0:
        return

    # words
    for block_idx, block in enumerate(transcription):
        match block:
            case Token():
                assert block.is_timed
                draw_line_with_ticks(
                    x1=block.start_time,
                    x2=block.end_time,
                    y=y_pos,
                    y_tick_width=y_tick_width,
                    ax=ax,
                    color='blue' if isinstance(block.value, Anything) else 'g',
                    lw=2,
                )
            case MultiVariant():
                for option_idx, option in enumerate(block.options):
                    option_y_pos = y_pos + y_delta * option_idx / (len(block.options) - 1)
                    for t in option:
                        draw_line_with_ticks(
                            x1=t.start_time,
                            x2=t.end_time,
                            y=option_y_pos,
                            y_tick_width=y_tick_width,
                            ax=ax,
                            color='blue' if isinstance(t.value, Anything) else 'g',
                            lw=2,
                        )
    
    # texts
    def write_token(token: Token, y: float):
        ax.text( # type: ignore
            (token.start_time + token.end_time) / 2,
            y,
            ' ' + str(token.value),
            fontsize=10,
            rotation=90,
            ha='center',
            va='bottom',
        )
    for block_idx, block in enumerate(transcription):
        match block:
            case Token():
                write_token(block, y_pos)
            case MultiVariant():
                best_option = sorted(
                    block.options,
                    key=lambda option: len(' '.join(str(t.value) for t in option))
                )[-1]
                for token in best_option:
                    write_token(token, y_pos)

    # bezier connections
    block_spans = [(x.start_time, x.end_time) for x in transcription]
    joints = (
        [block_spans[0][0] - 0.05]
        + [(end1 + start2) / 2  for (_, end1), (start2, _) in pairwise(block_spans)]
        + [block_spans[-1][1] + 0.05]
    )
    for block_idx, block in enumerate(transcription):
        prev_joint = joints[block_idx]
        next_joint = joints[block_idx + 1]
        match block:
            case Token():
                if block_idx > 0:
                    draw_bezier([
                        (prev_joint, y_pos),
                        (block.start_time, y_pos)
                    ], ax=ax, indent=0, lw=2)
                if block_idx < len(transcription) - 1:
                    draw_bezier([
                        (block.end_time, y_pos),
                        (next_joint, y_pos)
                    ], ax=ax, indent=0, lw=2)
            case MultiVariant():
                for option_idx, option in enumerate(block.options):
                    option_y_pos = y_pos + y_delta * option_idx / (len(block.options) - 1)
                    draw_bezier([
                        (prev_joint if option_idx == 0 else block.start_time - 0.05, y_pos),
                        (block.start_time, option_y_pos)
                    ], ax=ax, indent=0.05, lw=2)
                    draw_bezier([
                        (block.end_time, option_y_pos),
                        (next_joint if option_idx == 0 else block.end_time + 0.05, y_pos)
                    ], ax=ax, indent=0.05, lw=2)
                    draw_bezier([
                        (block.start_time, option_y_pos),
                        (block.end_time, option_y_pos)
                    ], ax=ax, indent=0, lw=2)

    # gray boxes
    if graybox_y is not None:
        for block_idx, block in enumerate(transcription):
            match block:
                case Token():
                    ax.fill_between( # type: ignore
                        [block.start_time, block.end_time],
                        [graybox_y[0], graybox_y[0]],
                        [graybox_y[1], graybox_y[1]],
                        color='#eeeeee',
                        zorder=-999,
                    )
                case MultiVariant():
                    for option_idx, option in enumerate(block.options):
                        for t in option:
                            ax.fill_between( # type: ignore
                                [t.start_time, t.end_time],
                                [graybox_y[0], graybox_y[0]],
                                [graybox_y[1], graybox_y[1]],
                                color='#eeeeee',
                                zorder=-999,
                            )