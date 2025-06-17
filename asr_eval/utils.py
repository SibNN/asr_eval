from datetime import timedelta
import uuid
from typing import TypeVar

import srt
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


T = TypeVar('T')

def N(val: T | None) -> T:
    """
    Checks that the variable is not None and returs it, useful to calm down type checker.
    """
    # https://discuss.python.org/t/improve-typing-with-to-force-not-none-value/7840/15
    assert val is not None
    return val


def new_uid() -> str:
    return str(uuid.uuid4())


def utterances_to_srt(utterances: list[tuple[str, float, float]]) -> str:
    '''
    Composes an SRT file.
    '''
    return srt.compose([ # type: ignore
        srt.Subtitle(
            index=None,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text
        )
        for text, start, end in utterances
    ], reindex=True)


# def extend_lims(
#     ax: plt.Axes, # pyright: ignore[reportPrivateImportUsage]
#     xmin: float | None = None,
#     xmax: float | None = None,
#     ymin: float | None = None,
#     ymax: float | None = None,
#     dxmin: float | None = None,
#     dxmax: float | None = None,
#     dymin: float | None = None,
#     dymax: float | None = None,
# ):
#     cur_xmin, cur_xmax = ax.get_xlim()
#     cur_ymin, cur_ymax = ax.get_ylim()
    
#     if xmin is not None:
#         cur_xmin = min(cur_xmin, xmin)
#     if xmax is not None:
#         cur_xmax = max(cur_xmax, xmax)
#     if ymin is not None:
#         cur_ymin = min(cur_ymin, ymin)
#     if ymax is not None:
#         cur_ymax = max(cur_ymax, ymax)
        
#     if dxmin is not None:
#         cur_xmin += dxmin
#     if dxmax is not None:
#         cur_xmax += dxmax
#     if dymin is not None:
#         cur_ymin += dymin
#     if dymax is not None:
#         cur_ymax += dymax
    
#     ax.set_xlim(cur_xmin, cur_xmax)
#     ax.set_ylim(cur_ymin, cur_ymax)

# def draw_horizontal_interval(
#     x1: float,
#     x2: float,
#     y: float,
#     color: str = 'C0',
#     lw: float = 0.2,
#     ax: plt.Axes | None = None # pyright: ignore[reportPrivateImportUsage]
# ):
#     ax = ax or plt.gca()
#     ax.add_patch(Rectangle(
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
#     ax: plt.Axes | None = None # pyright: ignore[reportPrivateImportUsage]
# ):
#     ax = ax or plt.gca()
#     plt.scatter([x], [y], s=s, c=color) # type: ignore
#     # extend_lims(xmin=x, xmax=x, ymin=y, ymax=y, ax=ax)