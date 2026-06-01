r"""Figure 1: Global map of hypoxic percentage per site, by decade."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.lines import Line2D
from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE
from coasty.visualize.const import (
    ALPHA_SCATTER,
    CMAP_HYPOXIA_PCT,
    FIGURE_DPI,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_MAP,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LEGEND_FRAMEALPHA,
    LINE_WIDTH_THIN,
)
from coasty.visualize.utils import (
    compute_hypoxic_flag,
    compute_site_stats,
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
        if k < len(N_HYPOXIC_THRESHOLDS):
            label = rf"$< {N_HYPOXIC_THRESHOLDS[k]}$"
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


def _make_category_map(
    ax: plt.Axes,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    pct: np.ndarray,
    n_hypoxic: np.ndarray,
    title: str,
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

    # Combined legend — outside axes, anchored to upper-right corner
    ax.legend(
        handles=_build_legend_handles(cat_colors),
        fontsize=FONT_SIZE_LEGEND,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        bbox_transform=ax.transAxes,
        framealpha=LEGEND_FRAMEALPHA,
        borderaxespad=0,
    )

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)


def _new_map_axes() -> tuple[plt.Figure, plt.Axes]:
    r"""Create a new figure with a Robinson-projection GeoAxes."""
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)
    return fig, ax


if __name__ == "__main__":
    #
    # Set math font to Computer Modern for LaTeX-style labels
    plt.rcParams["mathtext.fontset"] = "cm"

    # Loading data
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    is_hypoxic = compute_hypoxic_flag(ds)

    print(f"  Profiles: {len(lat):,}  |  Hypoxic: {is_hypoxic.sum():,}")

    # Time periods to plot: "all" + individual decades
    decades = list(range(1950, 2030, 10))
    time_periods = [("all", None)] + [(str(d), d) for d in decades]

    out_dir = Path(__file__).parent.parent / "plots" / "figure-1"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_fig = None

    for period_name, decade in time_periods:
        if decade is None:
            mask = np.ones(len(lat), dtype=bool)
            title_suffix = r"$1950$--present"
        else:
            mask = (years >= decade) & (years < decade + 10)
            title_suffix = rf"${decade}$s"

        if mask.sum() == 0:
            continue

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask], bin_size=BIN_SIZE)
        show = stats["n_hypoxic"] > 0

        if show.sum() == 0:
            continue

        s_lat = stats["lat"][show]
        s_lon = stats["lon"][show]
        pct = stats["pct_hypoxic"][show]
        n_hypoxic = stats["n_hypoxic"][show]

        mean_pct = stats["pct_hypoxic"].mean()
        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{len(stats['lat']):>5,} sites | {show.sum():>4,} hypoxic | "
            f"mean pct={mean_pct:.1f}%"
        )

        fig, ax = _new_map_axes()
        _make_category_map(
            ax,
            s_lat=s_lat,
            s_lon=s_lon,
            pct=pct,
            n_hypoxic=n_hypoxic,
            title=rf"Hypoxic percentage -- {title_suffix}",
        )
        fig.subplots_adjust(right=0.78)
        fig.savefig(
            out_dir / f"figure-1-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
