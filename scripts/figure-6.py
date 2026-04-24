import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import HYPOXIA_THRESHOLD
from coasty.visualize.const import (
    CMAP_OXYGEN_SEQUENTIAL,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_WIDE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FONT_SIZE_X_LABEL,
    FONT_SIZE_Y_LABEL,
    LINE_WIDTH,
    LINE_WIDTH_THICK,
)
from coasty.visualize.utils import (
    compute_bottom_oxygen,
    compute_site_median,
    plot_hypoxia_map,
)

DECADES = [
    (1960, "1960s", "steelblue"),
    (1980, "1980s", "darkorange"),
    (2010, "2010s", "crimson"),
]


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    bottom_o2 = compute_bottom_oxygen(ds)

    print(f"  Profiles with bottom O2: {np.isfinite(bottom_o2).sum():,} / {len(bottom_o2):,}")

    out_dir = Path(__file__).parent.parent / "plots" / "figure-6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Distribution figure ---
    fig_dist, ax_dist = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for decade, label, color in DECADES:
        mask = (years >= decade) & (years < decade + 10) & np.isfinite(bottom_o2)
        vals = bottom_o2[mask]

        if len(vals) < 10:
            print(f"  {label}: too few profiles, skipping")
            continue

        print(f"  {label}: {len(vals):,} profiles with bottom O2")

        hist_counts, hist_edges = np.histogram(vals, bins=80, density=True)
        hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        ax_dist.plot(
            hist_centers, hist_counts, color=color, linewidth=LINE_WIDTH_THICK, label=rf"${label}$"
        )
        ax_dist.fill_between(hist_centers, hist_counts, alpha=0.15, color=color)

    ax_dist.axvline(
        HYPOXIA_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=LINE_WIDTH,
        label=rf"Hypoxia threshold $({HYPOXIA_THRESHOLD:.0f}\ \mu$mol$/$kg$)$",
    )
    ax_dist.set_xlabel(r"Bottom $O_2$ $[\mu$mol$/$kg$]$", fontsize=FONT_SIZE_X_LABEL)
    ax_dist.set_ylabel(r"Probability density", fontsize=FONT_SIZE_Y_LABEL)
    ax_dist.set_title(
        r"Distribution of bottom $O_2$ -- three representative decades", fontsize=FONT_SIZE_TITLE
    )
    ax_dist.legend(fontsize=FONT_SIZE_LEGEND)
    ax_dist.tick_params(labelsize=FONT_SIZE_TICK)
    fig_dist.tight_layout(pad=1.5)
    fig_dist.savefig(
        out_dir / "figure-6-distributions.pdf", dpi=FIGURE_DPI_SAVE, bbox_inches="tight"
    )
    plt.close(fig_dist)

    # --- Maps: one per decade ---
    last_fig = None
    for decade, label, _ in DECADES:
        mask = (years >= decade) & (years < decade + 10) & np.isfinite(bottom_o2)

        if mask.sum() == 0:
            continue

        site_stats = compute_site_median(lat[mask], lon[mask], bottom_o2[mask])
        valid = np.isfinite(site_stats["median"])

        print(f"  {label} map: {valid.sum()} sites with median bottom O2")

        fig, _ = plot_hypoxia_map(
            site_lats=site_stats["lat"][valid],
            site_lons=site_stats["lon"][valid],
            metric=site_stats["median"][valid],
            title=rf"Median bottom $O_2$ -- ${label}$",
            cbar_label=r"Median bottom $O_2$ $[\mu$mol$/$kg$]$",
            cmap=CMAP_OXYGEN_SEQUENTIAL,
            log_scale=False,
        )
        fig.savefig(
            out_dir / f"figure-6-map-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
