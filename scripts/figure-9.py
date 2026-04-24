import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import (
    CMAP_SAMPLING_COUNT,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_WIDE,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FONT_SIZE_X_LABEL,
    FONT_SIZE_Y_LABEL,
)
from coasty.visualize.utils import plot_heatmap_2d

if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    print(f"  Profiles: {len(lat):,}")

    out_dir = Path(__file__).parent.parent / "plots" / "figure-9"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) Latitudinal histogram (1° bins) ---
    lat_edges = np.arange(-90, 91, 1)
    lat_counts, _ = np.histogram(lat, bins=lat_edges)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.bar(lat_centers, lat_counts, width=1.0, color="steelblue", alpha=0.8)
    ax.set_xlabel(r"Latitude $[°]$", fontsize=FONT_SIZE_X_LABEL)
    ax.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax.set_title(
        r"Latitudinal distribution of profiles ($1950$--present)", fontsize=FONT_SIZE_TITLE
    )
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "figure-9-lat.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    # --- (b) Longitudinal histogram (1° bins) ---
    lon_edges = np.arange(-180, 181, 1)
    lon_counts, _ = np.histogram(lon, bins=lon_edges)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.bar(lon_centers, lon_counts, width=1.0, color="darkorange", alpha=0.8)
    ax.set_xlabel(r"Longitude $[°]$", fontsize=FONT_SIZE_X_LABEL)
    ax.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax.set_title(
        r"Longitudinal distribution of profiles ($1950$--present)", fontsize=FONT_SIZE_TITLE
    )
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "figure-9-lon.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    # --- (c) 2-D heatmap (5° × 5° bins) ---
    lat5_edges = np.arange(-90, 91, 5)
    lon5_edges = np.arange(-180, 181, 5)
    lat5_centers = 0.5 * (lat5_edges[:-1] + lat5_edges[1:])
    lon5_centers = 0.5 * (lon5_edges[:-1] + lon5_edges[1:])

    count_2d, _, _ = np.histogram2d(lat, lon, bins=[lat5_edges, lon5_edges])
    count_2d = np.where(count_2d > 0, count_2d, np.nan)

    fig, _ = plot_heatmap_2d(
        lat_centers=lat5_centers,
        lon_centers=lon5_centers,
        values_2d=count_2d,
        title=r"Profile density per $5°\times5°$ cell ($1950$--present)",
        cbar_label=r"Number of profiles",
        cmap=CMAP_SAMPLING_COUNT,
    )
    fig.savefig(out_dir / "figure-9-heatmap.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.show()

    print(f"\nFigures saved to {out_dir}/")
