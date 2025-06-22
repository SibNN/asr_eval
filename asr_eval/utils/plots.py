import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches


from typing import Any


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


def draw_bezier(
    xy_points: list[tuple[float, float]],
    ax: plt.Axes,
    indent: float = 0.1,
    zorder: int = 0,
    lw: float = 1,
    color: str = 'darkgray',
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