from typing import Any
from itertools import pairwise

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.path

from .align.data import Token, MultiVariant


def draw_transcription(
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


def draw_bezier(
    xy_points: list[tuple[float, float]],
    ax: plt.Axes,
    indent: float = 0.1,
    zorder: int = 0,
    lw: float = 1,
    color: str = 'lightgray',
):
    verts: list[tuple[float, float]] = []
    for x, y in xy_points:
        for d in (-indent, 0, indent):
            verts.append((x + d, y))
    verts = verts[1:-1]
    codes = [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.CURVE4] * (len(verts) - 1)
    path = matplotlib.path.Path(verts, codes) # type: ignore
    patch = matplotlib.patches.PathPatch(
        path,
        facecolor='none',
        lw=lw,
        edgecolor=color,
        zorder=zorder
    )
    ax.add_patch(patch)
    # ax.autoscale()


def draw_line_with_ticks(
    x1: float,
    x2: float,
    y: float,
    y_tick_width: float,
    ax: plt.Axes,
    **kwargs: Any,
):
    ax.plot([x1, x2], [y, y], **kwargs,) # type: ignore
    ax.plot([x1, x1], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore
    ax.plot([x2, x2], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore


# def get_or_create_ax(figsize: tuple[float, float] = (6, 6)) -> plt.Axes:
#     if not plt.get_fignums():
#         plt.figure(figsize=figsize) # type: ignore
#     return plt.gca()


# def draw_horizontal_interval(
#     x1: float,
#     x2: float,
#     y: float,
#     color: str = 'C0',
#     lw: float = 0.2,
#     ax: plt.Axes | None = None
# ):
#     ax = ax or plt.gca()
#     ax.add_patch(matplotlib.patches.Rectangle(
#         (x1, y - lw / 2),
#         (x2 - x1),
#         lw,
#         linewidth=1,
#         color=color,
#     ))
#     # extend_lims(xmin=x1, xmax=x2, ymin=y - lw / 2, ymax=y + lw / 2, ax=ax)

# def draw_circle(
#     x: float,
#     y: float,
#     color: str = 'C0',
#     s: float = 10,
#     ax: plt.Axes | None = None
# ):
#     ax = ax or plt.gca()
#     plt.scatter([x], [y], s=s, c=color) # type: ignore
#     # extend_lims(xmin=x, xmax=x, ymin=y, ymax=y, ax=ax)