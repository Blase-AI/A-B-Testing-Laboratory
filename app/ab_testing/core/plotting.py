from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlotStyle:
    title_size: int = 12
    label_size: int = 10
    grid_alpha: float = 0.25
    dpi: int = 110


DEFAULT_STYLE = PlotStyle()


def new_figure(
    *,
    figsize: Tuple[float, float] = (10, 5),
    dpi: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi or style.dpi)
    ax.grid(True, alpha=style.grid_alpha)
    return fig, ax


def new_figure_grid(
    nrows: int,
    ncols: int,
    *,
    figsize: Tuple[float, float] = (12, 5),
    dpi: Optional[int] = None,
    style: PlotStyle = DEFAULT_STYLE,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi or style.dpi)
    for ax in axes.ravel():
        ax.grid(True, alpha=style.grid_alpha)
    return fig, axes

