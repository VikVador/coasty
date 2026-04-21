r"""Visualization utility functions for the Coasty project."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.figure import Figure

from .const import (
    ALPHA_SCATTER,
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    FIGURE_DPI,
    FIGURE_SIZE_MAP,
    FONT_SIZE_COLORBAR,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LINE_WIDTH_THIN,
    MARKER_SIZE_LARGE,
    MARKER_SIZE_SMALL,
)


def plot_hypoxia_map(
    site_lats: np.ndarray,
    site_lons: np.ndarray,
    metric: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str | plt.cm.ScalarMappable,
    log_scale: bool = False,
) -> tuple[Figure, plt.Axes]:
    r"""Create a global map with hypoxic sites sized and colored by a scalar metric.

    Arguments:
        - site_lats  : Latitudes of sites [°].
        - site_lons  : Longitudes of sites [°].
        - metric     : Values driving dot size and color (e.g. count or percentage).
        - title      : Figure title (supports LaTeX mathtext).
        - cbar_label : Colorbar label (supports LaTeX mathtext).
        - cmap       : Matplotlib colormap instance.
        - log_scale  : Apply log normalisation to dot sizes. Useful for count metrics
                       with heavy-tailed distributions. Default False.

    Returns:
        - fig, ax : Figure and Axes objects.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)

    # Normalise metric to [0, 1] for dot-size mapping
    vals = np.log1p(metric) if log_scale else metric.copy()
    vmin, vmax = vals.min(), vals.max()
    norm = (vals - vmin) / (vmax - vmin + 1e-10)
    sizes = (MARKER_SIZE_SMALL + norm * (MARKER_SIZE_LARGE - MARKER_SIZE_SMALL)) ** 2

    sc = ax.scatter(
        site_lons,
        site_lats,
        s=sizes,
        c=metric,
        cmap=cmap,
        alpha=ALPHA_SCATTER,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    cbar = fig.colorbar(
        sc, ax=ax, orientation="vertical", pad=COLORBAR_PAD, fraction=COLORBAR_FRACTION, shrink=0.8
    )
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)

    fig.tight_layout(pad=1.5)

    return fig, ax
