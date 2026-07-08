import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.lines import Line2D
from pathlib import Path

from coasty.config import PATH_DATASET, PATH_DATASET_DIAZ
from coasty.const import (BIN_SIZE, 
                          KM_PER_DEG,
                          MIN_CONSIDERED_YEAR,
                          MAX_CONSIDERED_YEAR)
from coasty.visualize.const import (
    CMAP_HYPOXIA_PCT,
    FIGURE_DPI_SAVE,
)
from coasty.visualize.hypoxia_maps import PCT_THRESHOLDS, make_category_map, new_map_axes
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

# ---------- Configurable parameters ----------
# Diaz overlay mode:
#   False → one triangle marker per Diaz site (raw, individual).
#   True  → one triangle per BIN_SIZE-km cell that contains ≥1 Diaz site (aggregated).
AGGREGATE_DIAZ = True

# Diaz triangle appearance.
DIAZ_MARKER_SIZE_PT2 = 40  # marker area [points²]
DIAZ_COLOR_FALLBACK = "red"


def _hypoxic_pct_colors() -> list:
    r"""Return the same color bins used for hypoxic-percentage circles."""
    n_cats = len(PCT_THRESHOLDS) + 1
    cmap_obj = plt.get_cmap(CMAP_HYPOXIA_PCT)
    return [cmap_obj(i / max(n_cats - 1, 1)) for i in range(n_cats)]


def _diaz_category_to_bin(hypoxia_current: str) -> int:
    r"""Map Diaz hypoxia_current category to the hypoxic-percentage bin index.

    Mapping requested:
      Diel -> <10
      Episodic -> 10--25
      Seasonal/Seasonl -> 25--50
      Periodic -> 50--75
      Persistent -> >=75
      Other/unknown -> fallback red
    """
    key = hypoxia_current.strip().lower()
    mapping = {
        "diel": 0,
        "episodic": 1,
        "seasonal": 2,
        "seasonl": 2,
        "periodic": 3,
        "persistent": 4,
    }
    return mapping.get(key, -1)


def aggregate_diaz_to_cells(
    diaz_lat: np.ndarray,
    diaz_lon: np.ndarray,
    diaz_bins: np.ndarray,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return one representative point per aggregation cell containing a Diaz site.

    Uses the same grid snapping as compute_site_stats so Diaz cells align perfectly
    with the ocean-profile aggregation grid.

    Arguments:
        - diaz_lat : Raw Diaz site latitudes [°].
        - diaz_lon : Raw Diaz site longitudes [°].
        - bin_size : Aggregation radius [km].

    Returns:
        - cell_lat, cell_lon : Deduplicated cell-centre coordinates [°].
        - cell_bin           : One category bin per cell (-1 uses fallback color).
    """
    bin_deg = bin_size / KM_PER_DEG
    snapped_lat = np.round(diaz_lat / bin_deg) * bin_deg
    snapped_lon = np.round(diaz_lon / bin_deg) * bin_deg
    unique_cells, inverse = np.unique(
        np.column_stack([snapped_lat, snapped_lon]), axis=0, return_inverse=True
    )

    # Keep one category per cell by taking the highest mapped bin among members.
    # This preserves persistent signals when mixed categories fall into one cell.
    cell_bin = np.full(len(unique_cells), -1, dtype=int)
    for idx in range(len(unique_cells)):
        bins_here = diaz_bins[inverse == idx]
        valid = bins_here[bins_here >= 0]
        if valid.size > 0:
            cell_bin[idx] = int(valid.max())

    return unique_cells[:, 0], unique_cells[:, 1], cell_bin


def overlay_diaz(
    ax: GeoAxes,
    diaz_lat: np.ndarray,
    diaz_lon: np.ndarray,
    diaz_hypoxia_current: np.ndarray,
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

    cat_colors = _hypoxic_pct_colors()
    diaz_bins = np.array([_diaz_category_to_bin(str(value)) for value in diaz_hypoxia_current])

    if AGGREGATE_DIAZ:
        plot_lat, plot_lon, plot_bins = aggregate_diaz_to_cells(
            diaz_lat,
            diaz_lon,
            diaz_bins,
            bin_size,
        )
    else:
        plot_lat, plot_lon, plot_bins = diaz_lat, diaz_lon, diaz_bins

    plot_colors = [cat_colors[k] if k >= 0 else DIAZ_COLOR_FALLBACK for k in plot_bins]

    ax.scatter(
        plot_lon,
        plot_lat,
        marker="^",
        s=DIAZ_MARKER_SIZE_PT2,
        color=plot_colors,
        edgecolors="black",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )


def _diaz_legend_handles() -> list[Line2D]:
    r"""Build Diaz triangle legend handles using the same color bins as pct legend."""
    cat_colors = _hypoxic_pct_colors()
    labels = [
        "Diaz Diel (<10)",
        "Diaz Episodic (10--25)",
        "Diaz Seasonal (25--50)",
        "Diaz Periodic (50--75)",
        "Diaz Persistent (>=75)",
        "Diaz Other",
    ]
    colors = [*cat_colors, DIAZ_COLOR_FALLBACK]

    return [
        Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markersize=np.sqrt(DIAZ_MARKER_SIZE_PT2),
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=label,
        )
        for label, color in zip(labels, colors)
    ]


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
    diaz_hypoxia_current = ds_diaz["hypoxia_current"].values.astype(str)

    # Apply a global Diaz filter in decade space based on considered-year bounds.
    # With MAX_CONSIDERED_YEAR=2025 this maps to last full decade 2020.
    min_diaz_decade = (MIN_CONSIDERED_YEAR // 10) * 10
    max_diaz_decade = (MAX_CONSIDERED_YEAR // 10) * 10
    diaz_in_window = (diaz_decade >= min_diaz_decade) & (diaz_decade <= max_diaz_decade)

    diaz_lat = diaz_lat[diaz_in_window]
    diaz_lon = diaz_lon[diaz_in_window]
    diaz_decade = diaz_decade[diaz_in_window]
    diaz_hypoxia_current = diaz_hypoxia_current[diaz_in_window]

    print(f"  Profiles: {len(lat):,}  |  Diaz sites: {len(diaz_lat)}")

    # --- Time periods ---
    decades = list(range(min_diaz_decade, max_diaz_decade + 10, 10))
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
        n_hypoxic = stats["n_hypoxic"][show]

        d_lat = diaz_lat[diaz_mask]
        d_lon = diaz_lon[diaz_mask]
        d_hypoxia_current = diaz_hypoxia_current[diaz_mask]

        mean_pct = stats["pct_hypoxic"].mean()
        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{show.sum():>4,} hypoxic sites | "
            f"{diaz_mask.sum():>3,} Diaz sites | mean pct={mean_pct:.1f}%"
        )

        fig, ax = new_map_axes()
        make_category_map(
            ax,
            s_lat=s_lat,
            s_lon=s_lon,
            pct=pct,
            n_hypoxic=n_hypoxic,
            title=rf"Hypoxic percentage + Diaz sites -- {title_suffix}",
            extra_legend_handles=_diaz_legend_handles(),
        )
        overlay_diaz(
            ax,
            d_lat,
            d_lon,
            d_hypoxia_current,
            bin_size=BIN_SIZE,
        )

        fig.subplots_adjust(right=0.78)
        fig.savefig(
            out_dir / f"figure-2-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
