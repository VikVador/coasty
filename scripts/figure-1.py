# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import CMAP_HYPOXIA_COUNT, CMAP_HYPOXIA_PCT, FIGURE_DPI_SAVE
from coasty.visualize.utils import plot_hypoxia_map

# CONSTANTS
HYPOXIA_THRESHOLD = 63.0  # µmol/kg
BIN_SIZE = 0.05  # degrees — ~5 km site grouping


# NEW FUNCTIONS
def compute_hypoxic_flag(ds: xr.Dataset) -> np.ndarray:
    r"""Return a boolean array indicating whether each profile is hypoxic.

    A profile is flagged hypoxic if at least one observation has DOX2 < 63 µmol/kg.

    Arguments:
        - ds : Loaded coasty dataset with variables DOX2 and profile_start.

    Returns:
        - is_hypoxic : Boolean array of shape (n_profiles,).
    """
    dox2 = ds["DOX2"].values.astype(float)
    np.nan_to_num(dox2, nan=np.inf, copy=False)  # NaN -> inf so they never trigger hypoxia

    profile_start = ds["profile_start"].values.astype(int) - 1  # convert to 0-indexed

    min_dox2 = np.minimum.reduceat(dox2, profile_start)

    return min_dox2 < HYPOXIA_THRESHOLD


def compute_site_stats(
    lat: np.ndarray,
    lon: np.ndarray,
    is_hypoxic: np.ndarray,
    bin_size: float = BIN_SIZE,
) -> dict:
    r"""Aggregate profiles into ~5 km sites and compute hypoxia statistics.

    Arguments:
        - lat        : Profile latitudes [°].
        - lon        : Profile longitudes [°].
        - is_hypoxic : Boolean hypoxic flag per profile.
        - bin_size   : Binning resolution in degrees (~5 km). Default 0.05.

    Returns:
        - stats : Dict with keys 'lat', 'lon', 'total', 'n_hypoxic', 'pct_hypoxic'.
    """
    site_lat = np.round(lat / bin_size) * bin_size
    site_lon = np.round(lon / bin_size) * bin_size

    # Encode each (lat, lon) bin as a unique integer — lon_idx < 8000 ensures no collision
    lat_idx = np.round((site_lat + 90.0) / bin_size).astype(int)
    lon_idx = np.round((site_lon + 180.0) / bin_size).astype(int)
    site_id = lat_idx * 8000 + lon_idx

    unique_ids, inverse = np.unique(site_id, return_inverse=True)

    total = np.bincount(inverse)
    n_hypoxic = np.bincount(inverse, weights=is_hypoxic.astype(float)).astype(int)
    pct = 100.0 * n_hypoxic / np.maximum(total, 1)

    out_lat = (unique_ids // 8000) * bin_size - 90.0
    out_lon = (unique_ids % 8000) * bin_size - 180.0

    return {
        "lat": out_lat,
        "lon": out_lon,
        "total": total,
        "n_hypoxic": n_hypoxic,
        "pct_hypoxic": pct,
    }


# MAIN CODE
if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    is_hypoxic = compute_hypoxic_flag(ds)

    print(f"  Profiles: {len(lat):,}  |  Hypoxic: {is_hypoxic.sum():,}")

    # --- Time periods: whole period then each decade ---
    decades = list(range(1950, 2030, 10))
    time_periods = [("all", None)] + [(str(d), d) for d in decades]

    # --- Output directory ---
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

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask])
        show = stats["n_hypoxic"] > 0

        if show.sum() == 0:
            continue

        s_lat = stats["lat"][show]
        s_lon = stats["lon"][show]
        count = stats["n_hypoxic"][show].astype(float)
        pct = stats["pct_hypoxic"][show]

        # Diagnostics: verify pct = (# hypoxic profiles / # total profiles at site) * 100
        n_total_sites = len(stats["lat"])
        mean_pct = pct.mean()
        median_pct = float(np.median(pct))
        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{n_total_sites:>6,} sites | {show.sum():>5,} hypoxic sites | "
            f"pct mean={mean_pct:.1f}% median={median_pct:.1f}%"
        )

        # Figure — dot size encodes number of hypoxic profiles
        fig, _ = plot_hypoxia_map(
            site_lats=s_lat,
            site_lons=s_lon,
            metric=count,
            title=rf"Hypoxic profiles -- {title_suffix}",
            cbar_label=r"Number of hypoxic profiles",
            cmap=CMAP_HYPOXIA_COUNT,
            log_scale=True,
        )
        fig.savefig(
            out_dir / f"figure-1-count-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Figure — dot size encodes hypoxic percentage
        fig, _ = plot_hypoxia_map(
            site_lats=s_lat,
            site_lons=s_lon,
            metric=pct,
            title=rf"Hypoxic percentage -- {title_suffix}",
            cbar_label=r"Hypoxic percentage $[\%]$",
            cmap=CMAP_HYPOXIA_PCT,
            log_scale=False,
        )
        fig.savefig(
            out_dir / f"figure-1-pct-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
