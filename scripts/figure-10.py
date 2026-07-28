import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET_SURFACE
from coasty.visualize.const import (
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_WIDE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    FONT_SIZE_X_LABEL,
    FONT_SIZE_Y_LABEL,
)
from coasty.visualize.utils import compute_hypoxic_flag

# PROMPT
figure_prompt = """

    Show the total number of ocean profiles collected per calendar month, aggregated across
    all years (1950–present) and all geographic locations.  Bars are split into non-hypoxic
    (blue) and hypoxic (orange) profiles so that both sampling effort and the seasonal
    hypoxia signal are visible in a single chart.

"""

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET_SURFACE, engine="netcdf4")
    months = ds["obs_time"].dt.month.values
    is_hypoxic = compute_hypoxic_flag(ds)

    all_months = np.arange(1, 13)

    # --- Count profiles per month ---
    total_per_month = np.array([(months == m).sum() for m in all_months], dtype=float)
    hypoxic_per_month = np.array(
        [(is_hypoxic[months == m]).sum() for m in all_months], dtype=float
    )
    non_hypoxic_per_month = total_per_month - hypoxic_per_month

    for m, tot, hyp in zip(all_months, total_per_month, hypoxic_per_month):
        print(
            f"  month={m:02d}: {int(tot):>7,} profiles | {int(hyp):>6,} hypoxic ({100 * hyp / tot:.1f}%)"
        )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    ax.bar(all_months, non_hypoxic_per_month, color="steelblue", alpha=0.8, label=r"Non-hypoxic")
    ax.bar(
        all_months,
        hypoxic_per_month,
        bottom=non_hypoxic_per_month,
        color="darkorange",
        alpha=0.85,
        label=r"Hypoxic",
    )

    ax.set_xticks(all_months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel(r"Month", fontsize=FONT_SIZE_X_LABEL)
    ax.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax.set_title(
        r"Monthly distribution of profiles ($1950$--present, all locations)",
        fontsize=FONT_SIZE_TITLE,
    )
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    fig.tight_layout(pad=1.5)

    out_path = Path(__file__).parent.parent / "plots" / "figure-10.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigure saved to {out_path}")
