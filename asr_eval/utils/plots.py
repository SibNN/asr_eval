from typing import Any

import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches


__all__ = [
    'draw_line_with_ticks',
    'draw_bezier',
]


def draw_line_with_ticks(
    x1: float,
    x2: float,
    y: float,
    y_tick_width: float,
    ax: plt.Axes,
    **kwargs: Any,
):
    '''
    Draws a line with ticks at the ends.
    '''
    ax.plot([x1, x2], [y, y], **kwargs,) # type: ignore
    ax.plot([x1, x1], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore
    ax.plot([x2, x2], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs,) # type: ignore


def draw_bezier(
    xy_points: list[tuple[float, float]],
    ax: plt.Axes,
    indent: float = 0.1,
    zorder: int = 0,
    lw: float = 1,
    color: str = 'darkgray',
):
    '''
    Draws a Bezier curve.
    '''
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