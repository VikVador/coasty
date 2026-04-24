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

DATATYPE_LABELS = {
    1: "Water bottle",
    2: "Argo float",
    3: "CTD",
}

DATATYPE_COLORS = {
    1: "steelblue",
    2: "darkorange",
    3: "seagreen",
}


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")
    months = ds["obs_time"].dt.month.values
    datatypes = ds["datatype"].values.astype(int)

    all_months = np.arange(1, 13)

    # --- Count profiles per month × datatype ---
    counts = {}
    for dt in DATATYPE_LABELS:
        dt_mask = datatypes == dt
        counts[dt] = np.array([(months[dt_mask] == m).sum() for m in all_months])
        print(f"  datatype {dt}: {dt_mask.sum():,} profiles total")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    bottom = np.zeros(12)
    for dt in DATATYPE_LABELS:
        ax.bar(
            all_months,
            counts[dt],
            bottom=bottom,
            color=DATATYPE_COLORS[dt],
            label=DATATYPE_LABELS[dt],
            alpha=0.85,
        )
        bottom += counts[dt]

    ax.set_xticks(all_months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=FONT_SIZE_TICK)
    ax.set_xlabel(r"Month", fontsize=FONT_SIZE_X_LABEL)
    ax.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax.set_title(r"Seasonal distribution of profiles by instrument type", fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    fig.tight_layout(pad=1.5)

    out_path = Path(__file__).parent.parent / "plots" / "figure-10.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigure saved to {out_path}")
