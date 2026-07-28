import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET_SURFACE
from coasty.visualize.const import (
    CMAP_HYPOXIA_PCT,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_WIDE,
    FONT_SIZE_COLORBAR,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FONT_SIZE_X_LABEL,
    FONT_SIZE_Y_LABEL,
)
from coasty.visualize.utils import compute_hypoxic_flag

# PROMPT
figure_prompt = """

    Show how the total number of ocean profiles is distributed across geographic coordinates,
    aggregated across all years (1950–present).  Produce two separate bar charts:
      (a) profile count per 1° latitude bin,
      (b) profile count per 1° longitude bin.
    Each bar is colored by the local hypoxic percentage using a diverging colormap,
    so both sampling effort and hypoxia intensity are visible in a single plot.

"""


def _colored_bar_chart(
    ax: plt.Axes,
    centers: np.ndarray,
    total: np.ndarray,
    pct: np.ndarray,
    width: float,
    cmap: mcolors.Colormap | str,
    xlabel: str,
    title: str,
) -> plt.cm.ScalarMappable:
    norm = mcolors.Normalize(vmin=0, vmax=100)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = cmap(norm(pct))
    ax.bar(centers, total, width=width, color=colors, edgecolor="none")
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_X_LABEL)
    ax.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET_SURFACE, engine="netcdf4")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    is_hypoxic = compute_hypoxic_flag(ds)

    print(f"  Profiles: {len(lat):,}  |  Hypoxic: {is_hypoxic.sum():,}")

    out_dir = Path(__file__).parent.parent / "plots" / "figure-9"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (a) Latitudinal histogram (1° bins) ---
    lat_edges = np.arange(-90, 91, 1)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    total_lat, _ = np.histogram(lat, bins=lat_edges)
    hypoxic_lat, _ = np.histogram(lat[is_hypoxic], bins=lat_edges)
    pct_lat = np.where(total_lat > 0, 100.0 * hypoxic_lat / total_lat, 0.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    sm = _colored_bar_chart(
        ax=ax,
        centers=lat_centers,
        total=total_lat.astype(float),
        pct=pct_lat,
        width=1.0,
        cmap=CMAP_HYPOXIA_PCT,
        xlabel=r"Latitude $[°]$",
        title=r"Latitudinal distribution of profiles ($1950$--present)",
    )
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"Hypoxic percentage $[\%]$", fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "figure-9-lat.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    # --- (b) Longitudinal histogram (1° bins) ---
    lon_edges = np.arange(-180, 181, 1)
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    total_lon, _ = np.histogram(lon, bins=lon_edges)
    hypoxic_lon, _ = np.histogram(lon[is_hypoxic], bins=lon_edges)
    pct_lon = np.where(total_lon > 0, 100.0 * hypoxic_lon / total_lon, 0.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    sm = _colored_bar_chart(
        ax=ax,
        centers=lon_centers,
        total=total_lon.astype(float),
        pct=pct_lon,
        width=1.0,
        cmap=CMAP_HYPOXIA_PCT,
        xlabel=r"Longitude $[°]$",
        title=r"Longitudinal distribution of profiles ($1950$--present)",
    )
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"Hypoxic percentage $[\%]$", fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "figure-9-lon.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.show()

    print(f"\nFigures saved to {out_dir}/")
