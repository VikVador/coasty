r"""Figure 1: Global map of hypoxic percentage per site, by decade."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from cartopy.mpl.geoaxes import GeoAxes
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from typing import cast

from coasty.const import HYPOXIA_THRESHOLD
from coasty.visualize.const import (
    ALPHA_SCATTER,
    CMAP_HYPOXIA_PCT,
    CMAP_OXYGEN_SEQUENTIAL,
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    FIGURE_DPI,
    FIGURE_SIZE_MAP,
    FONT_SIZE_COLORBAR,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LEGEND_FRAMEALPHA,
    LINE_WIDTH_THIN,
    MARKER_SIZE_SMALL,
)


# ==========
# PARAMETERS
# ==========
# Upper bounds for hypoxic-percentage color bins (4 thresholds → 5 color bins)
PCT_THRESHOLDS = [
    10,
    25,
    50,
    75,
]

# Upper bounds for n_hypoxic size bins (3 thresholds → 4 size bins)
N_HYPOXIC_THRESHOLDS = [
    10,
    100,
    500,
]

# Marker area [points²] for each size bin (must have len(N_HYPOXIC_THRESHOLDS) + 1 entries)
N_HYPOXIC_SIZES_PT2 = [
    10,
    50,
    100,
    200,
]

# Security check to ensure size bins are properly defined
assert len(N_HYPOXIC_SIZES_PT2) == len(N_HYPOXIC_THRESHOLDS) + 1, (
    f"N_HYPOXIC_SIZES_PT2 must have {len(N_HYPOXIC_THRESHOLDS) + 1} entries, "
    f"got {len(N_HYPOXIC_SIZES_PT2)}"
)


def _section_header(label: str) -> Line2D:
    r"""Return an invisible Line2D that renders as a text-only section header in a legend."""
    return Line2D([], [], linestyle="none", marker="none", markersize=0, label=label)


def _build_legend_handles(cat_colors: list) -> list[Line2D]:
    r"""Build combined legend handles with color and size sections separated by headers.

    Arguments:
        cat_colors : List of RGBA colors, one per percentage bin.

    Returns:
        handles : List of Line2D proxy handles — color section then size section.
    """
    fixed_markersize = np.sqrt(N_HYPOXIC_SIZES_PT2[1])

    # --- Color section (hypoxic percentage) ---
    color_handles = [_section_header(r"Hypoxic time $[\%]$")]
    n_cats = len(PCT_THRESHOLDS) + 1
    for k in range(n_cats):
        if k == 0:
            label = rf"$< {PCT_THRESHOLDS[0]}$"
        elif k < len(PCT_THRESHOLDS):
            label = rf"${PCT_THRESHOLDS[k - 1]}$--${PCT_THRESHOLDS[k]}$"
        else:
            label = rf"$\geq {PCT_THRESHOLDS[-1]}$"
        color_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=fixed_markersize,
                markerfacecolor=cat_colors[k],
                markeredgecolor="none",
                label=label,
            )
        )

    # --- Size section (hypoxic profile count) ---
    size_handles = [_section_header(r"# Profiles [-]")]
    n_bins = len(N_HYPOXIC_THRESHOLDS) + 1
    for k in range(n_bins):
        if k == 0:
            label = rf"$< {N_HYPOXIC_THRESHOLDS[0]}$"
        elif k < len(N_HYPOXIC_THRESHOLDS):
            label = rf"${N_HYPOXIC_THRESHOLDS[k - 1]}$--${N_HYPOXIC_THRESHOLDS[k]}$"
        else:
            label = rf"$\geq {N_HYPOXIC_THRESHOLDS[-1]}$"
        size_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=np.sqrt(N_HYPOXIC_SIZES_PT2[k]),
                markerfacecolor="gray",
                markeredgecolor="none",
                label=label,
            )
        )

    return [*color_handles, *size_handles]


def make_category_map(
    ax: GeoAxes,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    pct: np.ndarray,
    n_hypoxic: np.ndarray,
    title: str,
    extra_legend_handles: list[Line2D] | None = None,
) -> None:
    r"""Draw a bubble map on an existing GeoAxes with independent color and size encodings.

    Arguments:
        ax         : Cartopy GeoAxes to draw on.
        s_lat      : Site latitudes [°].
        s_lon      : Site longitudes [°].
        pct        : Hypoxic percentage per site [0–100] — drives dot color.
        n_hypoxic  : Number of hypoxic profiles per site — drives dot size.
        title      : Map title string (LaTeX mathtext).
    """
    n_pct_cats = len(PCT_THRESHOLDS) + 1
    cmap_obj = plt.get_cmap(CMAP_HYPOXIA_PCT)
    cat_colors = [cmap_obj(i / max(n_pct_cats - 1, 1)) for i in range(n_pct_cats)]

    # Color: bin each site by hypoxic percentage
    pct_bins = np.digitize(pct, PCT_THRESHOLDS)  # 0 … n_pct_cats-1
    colors = [cat_colors[k] for k in pct_bins]

    # Size: bin each site by number of hypoxic profiles
    n_bins = np.digitize(n_hypoxic, N_HYPOXIC_THRESHOLDS)  # 0 … len(N_HYPOXIC_THRESHOLDS)
    sizes = [N_HYPOXIC_SIZES_PT2[k] for k in n_bins]

    ax.scatter(
        s_lon,
        s_lat,
        s=sizes,
        c=colors,
        alpha=ALPHA_SCATTER,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    legend_handles = _build_legend_handles(cat_colors)
    if extra_legend_handles is not None:
        legend_handles.extend(extra_legend_handles)

    # Combined legend — outside axes, anchored to upper-right corner
    ax.legend(
        handles=legend_handles,
        fontsize=FONT_SIZE_LEGEND,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        bbox_transform=ax.transAxes,
        framealpha=LEGEND_FRAMEALPHA,
        borderaxespad=0,
    )

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)


def new_map_axes() -> tuple[Figure, GeoAxes]:
    r"""Create a new figure with a Robinson-projection GeoAxes."""
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = cast(GeoAxes, fig.add_subplot(1, 1, 1, projection=ccrs.Robinson()))
    ax.set_global()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)
    return fig, ax


def plot_oxygen_map(
    median_o2: np.ndarray,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    title: str,
    hypoxic_color: str = "red",
) -> Figure:
    r"""Create a world map of median deepest O2 with hypoxic sites highlighted in red.

    Non-hypoxic sites are colored by the sequential oxygen colormap; hypoxic sites
    (median O2 < HYPOXIA_THRESHOLD) are overplotted in red.  A dashed line on the
    colorbar marks the hypoxia threshold.

    Arguments:
        - median_o2 : Median deepest O2 per site [µmol/kg].
        - s_lat     : Site latitudes [°].
        - s_lon     : Site longitudes [°].
        - title     : Figure title (LaTeX mathtext).

    Returns:
        - fig : The created Figure.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = cast(GeoAxes, fig.add_subplot(1, 1, 1, projection=ccrs.Robinson()))
    ax.set_global()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)

    cmap_obj = plt.get_cmap(CMAP_OXYGEN_SEQUENTIAL)
    norm = mcolors.Normalize(vmin=np.nanmin(median_o2), vmax=np.nanmax(median_o2))

    hypoxic = median_o2 < HYPOXIA_THRESHOLD
    normal = ~hypoxic
    marker_s = MARKER_SIZE_SMALL**2

    # Normal sites — colored by oxygen concentration
    if normal.sum() > 0:
        ax.scatter(
            s_lon[normal],
            s_lat[normal],
            s=marker_s,
            c=median_o2[normal],
            cmap=cmap_obj,
            norm=norm,
            alpha=ALPHA_SCATTER,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    # Hypoxic sites — highlighted in red on top
    if hypoxic.sum() > 0:
        ax.scatter(
            s_lon[hypoxic],
            s_lat[hypoxic],
            s=marker_s,
            color=hypoxic_color,
            alpha=ALPHA_SCATTER,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    # Colorbar (represents the full oxygen range)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
        shrink=0.8,
    )
    cbar.set_label(
        r"Median deepest $O_2$ $[\mu\mathrm{mol\,kg}^{-1}]$",
        fontsize=FONT_SIZE_COLORBAR,
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Mark the hypoxia threshold on the colorbar
    cbar.ax.axhline(
        y=norm(HYPOXIA_THRESHOLD),
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    cbar.ax.text(
        1.15,
        norm(HYPOXIA_THRESHOLD),
        rf"${HYPOXIA_THRESHOLD:.0f}$",
        va="center",
        ha="left",
        fontsize=FONT_SIZE_TICK - 1,
        transform=cbar.ax.transAxes,
    )

    # Legend entry for hypoxic sites
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=MARKER_SIZE_SMALL,
                markerfacecolor=hypoxic_color,
                markeredgecolor="none",
                label=rf"Hypoxic ($< {HYPOXIA_THRESHOLD:.0f}\ \mu\mathrm{{mol\,kg}}^{{-1}}$)",
            )
        ],
        fontsize=FONT_SIZE_LEGEND,
        loc="lower left",
        framealpha=LEGEND_FRAMEALPHA,
    )

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    fig.tight_layout(pad=1.5)
    return fig
