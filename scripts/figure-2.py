import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET, PATH_DATASET_DIAZ
from coasty.visualize.const import (
    CMAP_HYPOXIA_COUNT,
    CMAP_HYPOXIA_PCT,
    DIAZ_COLOR_EPISODIC,
    DIAZ_COLOR_PERSISTENT,
    DIAZ_COLOR_SEASONAL,
    DIAZ_COLOR_UNKNOWN,
    FIGURE_DPI_SAVE,
    FONT_SIZE_LEGEND,
    MARKER_SIZE_LARGE,
)
from coasty.visualize.utils import (
    compute_hypoxic_flag,
    compute_site_stats,
    plot_hypoxia_map,
)

DIAZ_REGIME_COLORS = {
    "Seasonal": DIAZ_COLOR_SEASONAL,
    "Episodic": DIAZ_COLOR_EPISODIC,
    "Persistent": DIAZ_COLOR_PERSISTENT,
    "Unknown": DIAZ_COLOR_UNKNOWN,
}


def overlay_diaz(
    ax: plt.Axes, diaz_lat: np.ndarray, diaz_lon: np.ndarray, diaz_regime: np.ndarray
) -> None:
    r"""Overlay Diaz & Rosenberg hypoxic zones as colored stars on an existing map axes.

    Arguments:
        - ax          : Cartopy GeoAxes on which to draw.
        - diaz_lat    : Latitudes of Diaz sites [°].
        - diaz_lon    : Longitudes of Diaz sites [°].
        - diaz_regime : Normalized regime strings per site.
    """
    handles = []
    for regime, color in DIAZ_REGIME_COLORS.items():
        mask = diaz_regime == regime
        if mask.sum() == 0:
            continue
        sc = ax.scatter(
            diaz_lon[mask],
            diaz_lat[mask],
            marker="*",
            s=MARKER_SIZE_LARGE**2,
            color=color,
            edgecolors="black",
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
            zorder=4,
            label=rf"Diaz -- {regime}",
        )
        handles.append(sc)
    ax.legend(handles=handles, fontsize=FONT_SIZE_LEGEND, loc="lower left", framealpha=0.8)


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
    diaz_regime = ds_diaz["hypoxia_current"].to_series().str.strip().str.capitalize().values

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
        else:
            mask = (years >= decade) & (years < decade + 10)
            title_suffix = rf"${decade}$s"

        if mask.sum() == 0:
            continue

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask])
        show = stats["n_hypoxic"] > 0

        if show.sum() == 0:
            continue

        s_lat = stats["lat"][show]
        s_lon = stats["lon"][show]
        count = stats["n_hypoxic"][show].astype(float)
        pct = stats["pct_hypoxic"][show]

        print(f"  {period_name:>8s}: {mask.sum():>7,} profiles | {show.sum():>5,} hypoxic sites")

        # --- Count figure ---
        fig, ax = plot_hypoxia_map(
            site_lats=s_lat,
            site_lons=s_lon,
            metric=count,
            title=rf"Hypoxic profiles + Diaz sites -- {title_suffix}",
            cbar_label=r"Number of hypoxic profiles",
            cmap=CMAP_HYPOXIA_COUNT,
            log_scale=True,
        )
        overlay_diaz(ax, diaz_lat, diaz_lon, diaz_regime)
        fig.savefig(
            out_dir / f"figure-2-count-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        plt.close(fig)

        # --- Percentage figure ---
        fig, ax = plot_hypoxia_map(
            site_lats=s_lat,
            site_lons=s_lon,
            metric=pct,
            title=rf"Hypoxic percentage + Diaz sites -- {title_suffix}",
            cbar_label=r"Hypoxic percentage $[\%]$",
            cmap=CMAP_HYPOXIA_PCT,
            log_scale=False,
        )
        overlay_diaz(ax, diaz_lat, diaz_lon, diaz_regime)
        fig.savefig(
            out_dir / f"figure-2-pct-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
