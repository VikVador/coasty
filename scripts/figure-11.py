import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import KM_PER_DEG
from coasty.visualize.const import CMAP_COVERAGE, FIGURE_DPI_SAVE
from coasty.visualize.utils import plot_heatmap_2d

SPATIAL_BINS_KM = [5, 15, 50]
TEMPORAL_WINDOWS = [5, 10]


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    months = ds["obs_time"].dt.month.values

    out_dir = Path(__file__).parent.parent / "plots" / "figure-11"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_fig = None

    for bin_km in SPATIAL_BINS_KM:
        bin_deg = bin_km / KM_PER_DEG

        # Compute site_id for all profiles at this spatial resolution
        site_lat_arr = np.round(lat / bin_deg) * bin_deg
        site_lon_arr = np.round(lon / bin_deg) * bin_deg
        lat_idx = np.round((site_lat_arr + 90.0) / bin_deg).astype(int)
        lon_idx = np.round((site_lon_arr + 180.0) / bin_deg).astype(int)
        n_lon = int(np.ceil(360.0 / bin_deg)) + 1
        site_id = lat_idx * n_lon + lon_idx

        # Build a DataFrame for efficient groupby operations
        df = pd.DataFrame({"site_id": site_id, "year": years, "month": months})

        for window_yr in TEMPORAL_WINDOWS:
            print(f"  {bin_km} km, {window_yr} yr window...")

            df["block"] = (df["year"] - 1950) // window_yr

            # Count unique months per (site, block)
            month_coverage = df.groupby(["site_id", "block"])["month"].nunique()
            # For each site: max months covered in any block → best-case seasonal coverage
            max_months = month_coverage.groupby(level="site_id").max()

            frac_coverage = max_months / 12.0

            # Reconstruct lat/lon for each unique site
            unique_site_ids = max_months.index.values
            s_lat_idx = unique_site_ids // n_lon
            s_lon_idx = unique_site_ids % n_lon
            s_lat = s_lat_idx * bin_deg - 90.0
            s_lon = s_lon_idx * bin_deg - 180.0
            s_frac = frac_coverage.values

            print(
                f"    {len(s_lat):,} sites | "
                f"{(s_frac == 1.0).sum():,} fully covered ({100 * (s_frac == 1.0).mean():.1f}%)"
            )

            # Bin onto 5° × 5° heatmap for readability at global scale
            lat5_edges = np.arange(-90, 91, 5)
            lon5_edges = np.arange(-180, 181, 5)
            lat5_centers = 0.5 * (lat5_edges[:-1] + lat5_edges[1:])
            lon5_centers = 0.5 * (lon5_edges[:-1] + lon5_edges[1:])

            frac_sum, _, _ = np.histogram2d(
                s_lat, s_lon, bins=[lat5_edges, lon5_edges], weights=s_frac
            )
            cell_count, _, _ = np.histogram2d(s_lat, s_lon, bins=[lat5_edges, lon5_edges])
            mean_frac = np.where(cell_count > 0, frac_sum / cell_count, np.nan)

            fig, _ = plot_heatmap_2d(
                lat_centers=lat5_centers,
                lon_centers=lon5_centers,
                values_2d=mean_frac,
                title=(
                    rf"Seasonal coverage fraction -- " rf"${bin_km}$ km $\times$ ${window_yr}$ yr"
                ),
                cbar_label=r"Fraction of months covered $[0$--$1]$",
                cmap=CMAP_COVERAGE,
            )
            fname = f"figure-11-{bin_km}km-{window_yr}yr.pdf"
            fig.savefig(out_dir / fname, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
            last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
