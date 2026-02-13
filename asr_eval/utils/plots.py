from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.path
import matplotlib.patches
import numpy as np


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
    """Draws a horizontal line with ticks at the ends."""
    
    ax.plot([x1, x2], [y, y], **kwargs) # type: ignore
    ax.plot([x1, x1], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs) # type: ignore
    ax.plot([x2, x2], [y - y_tick_width / 2, y + y_tick_width / 2], **kwargs) # type: ignore
    

def draw_vline_with_ticks(
    x: float,
    y1: float,
    y2: float,
    x_tick_width: float,
    ax: plt.Axes, # type: ignore
    **kwargs: Any,
):
    """Draws a vertical line with ticks at the ends."""
    
    ax.plot([x, x], [y1, y2], **kwargs) # type: ignore
    ax.plot([x - x_tick_width / 2, x + x_tick_width / 2], [y1, y1], **kwargs) # type: ignore
    ax.plot([x - x_tick_width / 2, x + x_tick_width / 2], [y2, y2], **kwargs) # type: ignore


def draw_bezier(
    xy_points: list[tuple[float, float]],
    ax: plt.Axes,
    indent: float = 0.1,
    zorder: int = 0,
    lw: float = 1,
    color: str = 'darkgray',
):
    """Draws a Bezier curve."""
    
    verts: list[tuple[float, float]] = []
    for x, y in xy_points:
        for d in (-indent, 0, indent):
            verts.append((x + d, y))
    verts = verts[1:-1]
    codes = (
        [matplotlib.path.Path.MOVETO]
        + [matplotlib.path.Path.CURVE4] * (len(verts) - 1)
    )
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

def plot_length_histogram(
    lengths: list[float],
    save_path: str | Path | None = None,
    show: bool = True,
):
    """An utility to draw length histogram for audio samples."""
    
    _fig, ax = plt.subplots(figsize=(12, 6)) # type: ignore
    ax2 = ax.twinx()

    bins = np.logspace( # type: ignore
        np.log10(min(lengths)) - 0.2,
        np.log10(max(lengths)) + 0.2,
        num=200,
    )
    bin_centers = np.sqrt(bins[:-1] * bins[1:]) # type: ignore
    hist, _ = np.histogram(lengths, bins=bins) # type: ignore

    ax.hist(lengths, bins=bins, log=True, color='cornflowerblue') # type: ignore
    ax2.plot( # type: ignore
        bin_centers,
        np.cumsum(hist) / np.sum(hist),
        color='C1',
        label='cumulative ratio by samples'
    )
    ax2.plot( # type: ignore
        bin_centers,
        np.cumsum(hist * bin_centers) / np.sum(hist * bin_centers),
        color='red',
        label='cumulative ratio by seconds'
    )

    plt.xscale('log') # type: ignore
    plt.title('Audio length histogram and CDF') # type: ignore

    ax.set_xlabel('length (sec)') # type: ignore
    ax.set_ylabel('n samples') # type: ignore
    ax2.set_ylabel('ratio') # type: ignore
    plt.legend() # type: ignore

    ax2.set_yticks(np.linspace(0, 1, num=21)) # type: ignore
    ax2.grid(which='both', axis='y') # type: ignore
    ax.grid(which='both', axis='x') # type: ignore

    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(save_path)) # type: ignore
    if show:
        plt.show() # type: ignore
    plt.close()