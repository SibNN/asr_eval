from itertools import pairwise

import matplotlib.pyplot as plt

from asr_eval.align.data import MultiVariant, Token
from asr_eval.utils.plots import draw_bezier, draw_line_with_ticks


def draw_timed_transcription(
    transcription: list[Token | MultiVariant],
    y_pos: float = 0,
    y_delta: float = -1,
    y_tick_width: float = 0.1,
    ax: plt.Axes | None = None,
    graybox_y: tuple[float, float] | None = None,
):
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
                    color='g',
                    lw=2,
                )
            case MultiVariant():
                for option_idx, option in enumerate(block.options):
                    option_y_pos = y_pos + y_delta * option_idx
                    for t in option:
                        draw_line_with_ticks(
                            x1=t.start_time,
                            x2=t.end_time,
                            y=option_y_pos,
                            y_tick_width=y_tick_width,
                            ax=ax,
                            color='g',
                            lw=2,
                        )

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
                    option_y_pos = y_pos + y_delta * option_idx
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
                        zorder=-1,
                    )
                case MultiVariant():
                    for option_idx, option in enumerate(block.options):
                        for t in option:
                            ax.fill_between( # type: ignore
                                [t.start_time, t.end_time],
                                [graybox_y[0], graybox_y[0]],
                                [graybox_y[1], graybox_y[1]],
                                color='#eeeeee',
                                zorder=-1,
                            )