import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import (
    CMAP_HYPOXIA_COUNT,
    CMAP_HYPOXIA_PCT,
    FIGURE_DPI_SAVE,
)
from coasty.visualize.utils import (
    compute_hypoxic_flag,
    compute_site_stats,
    plot_heatmap_2d,
)

if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    is_hypoxic = compute_hypoxic_flag(ds)

    # --- Step 1: compute site-level stats at 5 km resolution ---
    stats = compute_site_stats(lat, lon, is_hypoxic)
    show = stats["n_hypoxic"] > 0
    s_lat = stats["lat"][show]
    s_lon = stats["lon"][show]
    s_count = stats["n_hypoxic"][show].astype(float)
    s_pct = stats["pct_hypoxic"][show]

    print(f"  Total hypoxic sites: {show.sum():,}")

    # --- Step 2: bin hypoxic sites onto 5° × 5° grid ---
    lat_edges = np.arange(-90, 91, 5)  # 37 edges → 36 cells
    lon_edges = np.arange(-180, 181, 5)  # 73 edges → 72 cells
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Count of hypoxic sites per 5° cell
    count_grid, _, _ = np.histogram2d(s_lat, s_lon, bins=[lat_edges, lon_edges])

    # Mean hypoxic percentage per 5° cell
    pct_sum_grid, _, _ = np.histogram2d(s_lat, s_lon, bins=[lat_edges, lon_edges], weights=s_pct)
    count_safe = np.where(count_grid > 0, count_grid, 1)  # avoid div-by-zero in np.where
    mean_pct_grid = np.where(count_grid > 0, pct_sum_grid / count_safe, np.nan)

    # Replace zeros in count with NaN for cleaner display
    count_grid = np.where(count_grid > 0, count_grid, np.nan)

    print(f"  5° grid: {(~np.isnan(count_grid)).sum()} non-empty cells")

    out_dir = Path(__file__).parent.parent / "plots" / "figure-5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Count figure ---
    fig, _ = plot_heatmap_2d(
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        values_2d=count_grid,
        title=r"Number of hypoxic sites per $5°\times5°$ cell ($1950$--present)",
        cbar_label=r"Number of hypoxic sites",
        cmap=CMAP_HYPOXIA_COUNT,
    )
    fig.savefig(out_dir / "figure-5-count.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    # --- Percentage figure ---
    fig, _ = plot_heatmap_2d(
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        values_2d=mean_pct_grid,
        title=r"Mean hypoxic percentage per $5°\times5°$ cell ($1950$--present)",
        cbar_label=r"Mean hypoxic percentage $[\%]$",
        cmap=CMAP_HYPOXIA_PCT,
    )
    fig.savefig(out_dir / "figure-5-pct.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.show()

    print(f"\nFigures saved to {out_dir}/")
