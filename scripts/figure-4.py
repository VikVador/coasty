import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE
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

# PROMPT
figure_prompt = """

    Show the climatological seasonal cycle of ocean sampling for the Northern and Southern
    Hemisphere separately.  For each hemisphere and each month show:
      (a) total number of observed profiles (stacked bars: non-hypoxic + hypoxic),
      (b) mean hypoxic percentage on a secondary y-axis (0–100 %).
    Mean hypoxic percentage is computed over ALL sites to avoid bias.

"""

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
        "Northern Hemisphere": lat >= 0,
        "Southern Hemisphere": lat < 0,
    }

    fig, axes = plt.subplots(
        2, 1, figsize=(FIGURE_SIZE_WIDE[0], FIGURE_SIZE_WIDE[1] * 2), sharex=True
    )

    # Compute y-limits for profile counts so both panels share the same scale
    all_totals = []
    for hemi_mask in hemi_masks.values():
        for m in range(1, 13):
            all_totals.append(int((hemi_mask & (months == m)).sum()))
    y_max_profiles = max(all_totals) * 1.15

    months_x = np.arange(1, 13)

    for ax, (hemi_label, hemi_mask) in zip(axes, hemi_masks.items()):
        n_obs_per_month = np.zeros(12, dtype=float)
        n_hypoxic_per_month = np.zeros(12, dtype=float)
        mean_pct_per_month = np.zeros(12, dtype=float)

        for m in range(1, 13):
            mask = hemi_mask & (months == m)
            total = int(mask.sum())
            hypoxic_count = int(is_hypoxic[mask].sum())

            n_obs_per_month[m - 1] = total
            n_hypoxic_per_month[m - 1] = hypoxic_count

            if total > 0:
                stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask], bin_size=BIN_SIZE)
                # Mean over ALL sites to avoid inflation from single-profile sites
                mean_pct_per_month[m - 1] = stats["pct_hypoxic"].mean()
            else:
                mean_pct_per_month[m - 1] = 0.0

            print(
                f"  {hemi_label[:1]}H  month={m:02d}: {total:>6,} profiles | "
                f"{hypoxic_count:>5,} hypoxic | mean pct={mean_pct_per_month[m - 1]:.1f}%"
            )

        n_non_hypoxic = n_obs_per_month - n_hypoxic_per_month

        # Stacked bars: non-hypoxic base + hypoxic on top
        ax.bar(
            months_x, n_non_hypoxic, color="steelblue", alpha=0.7, label=r"Non-hypoxic profiles"
        )
        ax.bar(
            months_x,
            n_hypoxic_per_month,
            bottom=n_non_hypoxic,
            color="darkorange",
            alpha=0.85,
            label=r"Hypoxic profiles",
        )
        ax.set_ylabel(r"Number of observed profiles", fontsize=FONT_SIZE_Y_LABEL, color="black")
        ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
        ax.set_ylim(0, y_max_profiles)

        # Secondary y-axis for mean hypoxic %
        ax2 = ax.twinx()
        ax2.plot(
            months_x,
            mean_pct_per_month,
            color="crimson",
            linewidth=LINE_WIDTH_THICK,
            marker="o",
            markersize=5,
            label=r"Mean hypoxic $[\%]$",
        )
        ax2.set_ylabel(
            r"Mean hypoxic percentage $[\%]$", fontsize=FONT_SIZE_Y_LABEL, color="crimson"
        )
        ax2.tick_params(axis="y", labelcolor="crimson", labelsize=FONT_SIZE_TICK)
        ax2.set_ylim(0, 100)

        ax.set_title(rf"{hemi_label} -- climatological seasonal cycle", fontsize=FONT_SIZE_TITLE)
        ax.set_xticks(months_x)
        ax.set_xticklabels(MONTH_LABELS, fontsize=FONT_SIZE_TICK)

        # Combined legend in upper right
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
