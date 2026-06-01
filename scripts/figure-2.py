import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.lines import Line2D
from pathlib import Path

from coasty.config import PATH_DATASET, PATH_DATASET_DIAZ
from coasty.const import BIN_SIZE, KM_PER_DEG
from coasty.visualize.const import (
    ALPHA_SCATTER,
    CMAP_HYPOXIA_PCT,
    DIAZ_COLOR_PERSISTENT,
    FIGURE_DPI,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_MAP,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LEGEND_FRAMEALPHA,
    LINE_WIDTH_THIN,
)
from coasty.visualize.utils import compute_hypoxic_flag, compute_site_stats

# PROMPT
figure_prompt = """

    Show the spatial distribution of hypoxic profiles worldwide overlaid with Diaz &
    Rosenberg (2008) historical hypoxic zones, for the full 1950–present period and for each
    decade separately.  Circle size encodes the hypoxic percentage per site in discrete bins
    defined by CIRCLE_THRESHOLDS.  Diaz sites are shown as triangles, either as raw individual
    markers (AGGREGATE_DIAZ = False) or aggregated into BIN_SIZE-km cells with one triangle per
    occupied cell (AGGREGATE_DIAZ = True).  The full legend is shown on every plot.
    Aggregation radius is controlled by BIN_SIZE in coasty/const.py.

"""

# ── Configurable parameters ──────────────────────────────────────────────────
# Upper bounds for each circle-size bin [hypoxic percentage, 0–100].
# A final "≥ last threshold" bin is added automatically.
CIRCLE_THRESHOLDS = [10, 25, 50, 75]

# Marker area [points²] for each bin (must have len(CIRCLE_THRESHOLDS) + 1 entries).
CIRCLE_SIZES_PT2 = [20, 40, 60, 80, 100]

# Diaz overlay mode:
#   False → one triangle marker per Diaz site (raw, individual).
#   True  → one triangle per BIN_SIZE-km cell that contains ≥1 Diaz site (aggregated).
AGGREGATE_DIAZ = True

# Diaz triangle appearance.
DIAZ_COLOR = DIAZ_COLOR_PERSISTENT
DIAZ_MARKER_SIZE_PT2 = 40  # marker area [points²]
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_diaz_to_cells(
    diaz_lat: np.ndarray,
    diaz_lon: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return one representative point per aggregation cell containing a Diaz site.

    Uses the same grid snapping as compute_site_stats so Diaz cells align perfectly
    with the ocean-profile aggregation grid.

    Arguments:
        - diaz_lat : Raw Diaz site latitudes [°].
        - diaz_lon : Raw Diaz site longitudes [°].
        - bin_size : Aggregation radius [km].

    Returns:
        - cell_lat, cell_lon : Deduplicated cell-centre coordinates [°].
    """
    bin_deg = bin_size / KM_PER_DEG
    snapped_lat = np.round(diaz_lat / bin_deg) * bin_deg
    snapped_lon = np.round(diaz_lon / bin_deg) * bin_deg
    unique_cells = np.unique(np.column_stack([snapped_lat, snapped_lon]), axis=0)
    return unique_cells[:, 0], unique_cells[:, 1]


def overlay_diaz(
    ax: plt.Axes,
    diaz_lat: np.ndarray,
    diaz_lon: np.ndarray,
    bin_size: float,
) -> None:
    r"""Overlay Diaz & Rosenberg sites as triangles on an existing map axes.

    Behaviour is controlled by the AGGREGATE_DIAZ flag:
      - False : one triangle per Diaz site (raw positions).
      - True  : one triangle per BIN_SIZE-km cell that contains at least one site.

    Arguments:
        - ax       : Cartopy GeoAxes to draw on.
        - diaz_lat : Diaz site latitudes for the current period [°].
        - diaz_lon : Diaz site longitudes for the current period [°].
        - bin_size : Aggregation radius used for the main dataset [km].
    """
    if len(diaz_lat) == 0:
        return

    if AGGREGATE_DIAZ:
        plot_lat, plot_lon = aggregate_diaz_to_cells(diaz_lat, diaz_lon, bin_size)
    else:
        plot_lat, plot_lon = diaz_lat, diaz_lon

    ax.scatter(
        plot_lon,
        plot_lat,
        marker="^",
        s=DIAZ_MARKER_SIZE_PT2,
        color=DIAZ_COLOR,
        edgecolors="black",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )


def _build_legend_handles(cat_colors: list) -> list[Line2D]:
    r"""Build legend handles for ALL percentage categories plus the Diaz triangle.

    The full list is always returned so the legend is identical across all time-
    period plots, making them directly comparable.

    Arguments:
        - cat_colors : List of RGBA colors, one per category.

    Returns:
        - handles : List of Line2D proxy handles.
    """
    n_cats = len(CIRCLE_THRESHOLDS) + 1
    handles = []
    for k in range(n_cats):
        if k == 0:
            label = rf"$< {CIRCLE_THRESHOLDS[0]}\,\%$"
        elif k < len(CIRCLE_THRESHOLDS):
            label = rf"${CIRCLE_THRESHOLDS[k - 1]}$--${CIRCLE_THRESHOLDS[k]}\,\%$"
        else:
            label = rf"$\geq {CIRCLE_THRESHOLDS[-1]}\,\%$"

        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=np.sqrt(CIRCLE_SIZES_PT2[k]),
                markerfacecolor=cat_colors[k],
                markeredgecolor="none",
                label=label,
            )
        )

    diaz_label = r"Diaz \& Rosenberg (aggregated)" if AGGREGATE_DIAZ else r"Diaz \& Rosenberg"
    handles.append(
        Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markersize=np.sqrt(DIAZ_MARKER_SIZE_PT2),
            markerfacecolor=DIAZ_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=diaz_label,
        )
    )
    return handles


def _make_category_map(
    ax: plt.Axes,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    pct: np.ndarray,
    diaz_lat: np.ndarray,
    diaz_lon: np.ndarray,
    title: str,
) -> None:
    r"""Draw a discrete-size bubble map with Diaz overlay on an existing GeoAxes.

    Arguments:
        - ax       : Cartopy GeoAxes to draw on.
        - s_lat    : Site latitudes [°].
        - s_lon    : Site longitudes [°].
        - pct      : Hypoxic percentage per site [0–100].
        - diaz_lat : Diaz site latitudes for the current period [°].
        - diaz_lon : Diaz site longitudes for the current period [°].
        - title    : Map title string (LaTeX mathtext).
    """
    n_cats = len(CIRCLE_THRESHOLDS) + 1
    cmap_obj = plt.get_cmap(CMAP_HYPOXIA_PCT)
    cat_colors = [cmap_obj(i / max(n_cats - 1, 1)) for i in range(n_cats)]

    categories = np.digitize(pct, CIRCLE_THRESHOLDS)

    for k in range(n_cats):
        mask = categories == k
        if mask.sum() == 0:
            continue
        ax.scatter(
            s_lon[mask],
            s_lat[mask],
            s=CIRCLE_SIZES_PT2[k],
            color=cat_colors[k],
            alpha=ALPHA_SCATTER,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    overlay_diaz(ax, diaz_lat, diaz_lon, bin_size=BIN_SIZE)

    ax.legend(
        handles=_build_legend_handles(cat_colors),
        fontsize=FONT_SIZE_LEGEND,
        loc="lower left",
        framealpha=LEGEND_FRAMEALPHA,
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
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load main dataset ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    is_hypoxic = compute_hypoxic_flag(ds)

    # --- Load Diaz dataset ---
    ds_diaz = xr.open_zarr(PATH_DATASET_DIAZ)
    diaz_lat = ds_diaz["lat"].values
    diaz_lon = ds_diaz["lon"].values
    diaz_decade = ds_diaz["decade"].values.astype(int)

    print(f"  Profiles: {len(lat):,}  |  Diaz sites: {len(diaz_lat)}")

    # --- Time periods ---
    decades = list(range(1950, 2030, 10))
    time_periods = [("all", None)] + [(str(d), d) for d in decades]

    out_dir = Path(__file__).parent.parent / "plots" / "figure-2"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_fig = None

    for period_name, decade in time_periods:
        if decade is None:
            mask = np.ones(len(lat), dtype=bool)
            title_suffix = r"$1950$--present"
            diaz_mask = np.ones(len(diaz_lat), dtype=bool)
        else:
            mask = (years >= decade) & (years < decade + 10)
            title_suffix = rf"${decade}$s"
            diaz_mask = diaz_decade == decade

        if mask.sum() == 0:
            continue

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask], bin_size=BIN_SIZE)
        show = stats["n_hypoxic"] > 0

        if show.sum() == 0:
            continue

        s_lat = stats["lat"][show]
        s_lon = stats["lon"][show]
        pct = stats["pct_hypoxic"][show]

        d_lat = diaz_lat[diaz_mask]
        d_lon = diaz_lon[diaz_mask]

        mean_pct = stats["pct_hypoxic"].mean()
        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{show.sum():>4,} hypoxic sites | "
            f"{diaz_mask.sum():>3,} Diaz sites | mean pct={mean_pct:.1f}%"
        )

        fig, ax = _new_map_axes()
        _make_category_map(
            ax,
            s_lat=s_lat,
            s_lon=s_lon,
            pct=pct,
            diaz_lat=d_lat,
            diaz_lon=d_lon,
            title=rf"Hypoxic percentage + Diaz sites -- {title_suffix}",
        )
        fig.tight_layout(pad=1.5)
        fig.savefig(
            out_dir / f"figure-2-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
