import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import (
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_WIDE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FONT_SIZE_X_LABEL,
    FONT_SIZE_Y_LABEL,
    LINE_WIDTH_THICK,
)
from coasty.visualize.utils import (
    compute_hypoxic_flag,
    compute_site_stats,
)

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    months = ds["obs_time"].dt.month.values
    is_hypoxic = compute_hypoxic_flag(ds)

    hemi_masks = {
        "Northern Hemisphere": lat > 0,
        "Southern Hemisphere": lat < 0,
    }

    fig, axes = plt.subplots(
        2, 1, figsize=(FIGURE_SIZE_WIDE[0], FIGURE_SIZE_WIDE[1] * 2), sharex=True
    )

    for ax, (hemi_label, hemi_mask) in zip(axes, hemi_masks.items()):
        n_sites_per_month = np.zeros(12)
        mean_pct_per_month = np.zeros(12)

        for m in range(1, 13):
            mask = hemi_mask & (months == m)
            if mask.sum() == 0:
                n_sites_per_month[m - 1] = 0
                mean_pct_per_month[m - 1] = 0
                continue

            stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask])
            hypoxic = stats["n_hypoxic"] > 0
            n_sites_per_month[m - 1] = hypoxic.sum()
            mean_pct_per_month[m - 1] = (
                stats["pct_hypoxic"][hypoxic].mean() if hypoxic.sum() > 0 else 0.0
            )

            print(
                f"  {hemi_label[:1]}H  month={m:02d}: {mask.sum():>6,} profiles | {hypoxic.sum():>4,} hypoxic sites"
            )

        months_x = np.arange(1, 13)

        # Bar chart for site count
        ax.bar(months_x, n_sites_per_month, color="steelblue", alpha=0.6, label=r"Hypoxic sites")
        ax.set_ylabel(r"Number of hypoxic sites", fontsize=FONT_SIZE_Y_LABEL, color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue", labelsize=FONT_SIZE_TICK)

        # Secondary y-axis for mean pct
        ax2 = ax.twinx()
        ax2.plot(
            months_x,
            mean_pct_per_month,
            color="darkorange",
            linewidth=LINE_WIDTH_THICK,
            marker="o",
            markersize=5,
            label=r"Mean hypoxic $[\%]$",
        )
        ax2.set_ylabel(
            r"Mean hypoxic percentage $[\%]$", fontsize=FONT_SIZE_Y_LABEL, color="darkorange"
        )
        ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=FONT_SIZE_TICK)

        ax.set_title(rf"{hemi_label} -- climatological seasonal cycle", fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(months_x)
        ax.set_xticklabels(MONTH_LABELS, fontsize=FONT_SIZE_TICK)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_SIZE_LEGEND, loc="upper right")

    axes[-1].set_xlabel(r"Month", fontsize=FONT_SIZE_X_LABEL)
    fig.tight_layout(pad=1.5)

    out_path = Path(__file__).parent.parent / "plots" / "figure-4.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigure saved to {out_path}")
