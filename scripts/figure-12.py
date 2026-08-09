import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import (
    ALPHA_SCATTER,
    FIGURE_DPI,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_MAP,
    FIGURE_SIZE_WIDE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FONT_SIZE_X_LABEL,
    FONT_SIZE_Y_LABEL,
    LINE_WIDTH_THIN,
    MARKER_SIZE,
    SAMPLING_COLOR_EPISODIC,
    SAMPLING_COLOR_INTENSE,
    SAMPLING_COLOR_IRREGULAR,
    SAMPLING_COLOR_MODERATE,
)
from coasty.visualize.utils import compute_site_stats

CATEGORIES = [
    (
        "Episodic",
        SAMPLING_COLOR_EPISODIC,
        r"Episodic ($< 5$)",
    ),
    (
        "Irregular",
        SAMPLING_COLOR_IRREGULAR,
        r"Irregular ($5$--$10$)",
    ),
    (
        "Moderate",
        SAMPLING_COLOR_MODERATE,
        r"Moderate ($10$--$20$)",
    ),
    (
        "Intense",
        SAMPLING_COLOR_INTENSE,
        r"Intense ($> 20$)",
    ),
]


def classify_intensity(profiles_per_decade: np.ndarray) -> np.ndarray:
    r"""Classify sites by profiles-per-decade into four intensity categories.

    Arguments:
        - profiles_per_decade : Mean profiles per sampled decade per site.

    Returns:
        - labels : String category label per site.
    """
    labels = np.empty(len(profiles_per_decade), dtype=object)
    labels[profiles_per_decade < 5] = "Episodic"
    labels[(profiles_per_decade >= 5) & (profiles_per_decade < 10)] = "Irregular"
    labels[(profiles_per_decade >= 10) & (profiles_per_decade <= 20)] = "Moderate"
    labels[profiles_per_decade > 20] = "Intense"
    return labels


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values

    is_hypoxic_dummy = np.zeros(len(lat), dtype=bool)

    # --- Compute total profiles and decades sampled per site ---
    stats = compute_site_stats(lat, lon, is_hypoxic_dummy)

    # Count distinct decades sampled per site
    # Use same binning logic to get per-site decade counts
    from coasty.const import BIN_SIZE, KM_PER_DEG

    bin_deg = BIN_SIZE / KM_PER_DEG
    site_lat = np.round(lat / bin_deg) * bin_deg
    site_lon = np.round(lon / bin_deg) * bin_deg
    lat_idx = np.round((site_lat + 90.0) / bin_deg).astype(int)
    lon_idx = np.round((site_lon + 180.0) / bin_deg).astype(int)
    n_lon = int(np.ceil(360.0 / bin_deg)) + 1
    site_id = lat_idx * n_lon + lon_idx

    decade_of_profile = (years // 10) * 10

    unique_ids, inverse = np.unique(site_id, return_inverse=True)

    # Count distinct decades per site
    n_decades = np.array([
        len(np.unique(decade_of_profile[inverse == k])) for k in range(len(unique_ids))
    ])

    total = stats["total"]
    profiles_per_dec = total / np.maximum(n_decades, 1).astype(float)
    labels = classify_intensity(profiles_per_dec)
    site_lats = stats["lat"]
    site_lons = stats["lon"]

    print(f"  Total sites: {len(site_lats):,}")
    for cat, _, _ in CATEGORIES:
        print(f"    {cat}: {(labels == cat).sum():,}")

    out_dir = Path(__file__).parent.parent / "plots" / "figure-12"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Global map ---
    fig_map = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax_map.set_global()
    fig_map.patch.set_facecolor("white")
    ax_map.set_facecolor("white")
    ax_map.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax_map.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)

    handles = []
    for cat_name, color, cat_label in CATEGORIES:
        mask = labels == cat_name
        if mask.sum() == 0:
            continue
        ax_map.scatter(
            site_lons[mask],
            site_lats[mask],
            s=MARKER_SIZE**2,
            color=color,
            alpha=ALPHA_SCATTER,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        handles.append(mpatches.Patch(color=color, label=cat_label))

    ax_map.legend(handles=handles, fontsize=FONT_SIZE_LEGEND, loc="lower left", framealpha=0.8)
    ax_map.set_title(r"Sampling intensity classification per site", fontsize=FONT_SIZE_TITLE)
    fig_map.tight_layout(pad=1.5)
    fig_map.savefig(out_dir / "figure-12-map.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
    plt.close(fig_map)

    # --- Bar chart ---
    cat_names = [c[0] for c in CATEGORIES]
    cat_colors = [c[1] for c in CATEGORIES]
    cat_labels = [c[2] for c in CATEGORIES]
    cat_counts = [(labels == c).sum() for c in cat_names]

    fig_bar, ax_bar = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    bars = ax_bar.bar(
        cat_names, cat_counts, color=cat_colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )

    for bar, count in zip(bars, cat_counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cat_counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_TICK,
        )

    ax_bar.set_xticks(range(len(cat_names)))
    ax_bar.set_xticklabels(cat_labels, fontsize=FONT_SIZE_TICK)
    ax_bar.set_ylabel(r"Number of sites", fontsize=FONT_SIZE_Y_LABEL)
    ax_bar.set_xlabel(r"Sampling intensity category", fontsize=FONT_SIZE_X_LABEL)
    ax_bar.set_title(
        r"Site count by sampling intensity (profiles per decade)", fontsize=FONT_SIZE_TITLE
    )
    ax_bar.tick_params(labelsize=FONT_SIZE_TICK)
    fig_bar.tight_layout(pad=1.5)
    fig_bar.savefig(out_dir / "figure-12-bar.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigures saved to {out_dir}/")
