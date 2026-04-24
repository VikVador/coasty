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

    years = ds["obs_time"].dt.year.values
    datatypes = ds["datatype"].values.astype(int)

    all_years = np.arange(years.min(), years.max() + 1)

    # --- Count profiles per year × datatype ---
    counts = {}
    for dt in DATATYPE_LABELS:
        dt_mask = datatypes == dt
        counts[dt] = np.array([(years[dt_mask] == yr).sum() for yr in all_years])
        print(f"  datatype {dt}: {dt_mask.sum():,} profiles total")

    total_per_year = sum(counts.values())
    cumulative = np.cumsum(total_per_year)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIGURE_SIZE_WIDE[0], FIGURE_SIZE_WIDE[1] * 2), sharex=True
    )

    # Panel (a) — stacked bars
    bottom = np.zeros(len(all_years))
    for dt in DATATYPE_LABELS:
        ax1.bar(
            all_years,
            counts[dt],
            bottom=bottom,
            color=DATATYPE_COLORS[dt],
            label=DATATYPE_LABELS[dt],
            alpha=0.85,
        )
        bottom += counts[dt]

    ax1.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax1.set_title(
        r"Sampling effort -- annual profile count by instrument type ($1950$--present)",
        fontsize=FONT_SIZE_TITLE,
    )
    ax1.legend(fontsize=FONT_SIZE_LEGEND)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)

    # Panel (b) — cumulative
    ax2.plot(all_years, cumulative, color="navy", linewidth=LINE_WIDTH_THICK)
    ax2.fill_between(all_years, cumulative, alpha=0.15, color="navy")
    ax2.set_ylabel(r"Cumulative profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax2.set_xlabel(r"Year", fontsize=FONT_SIZE_X_LABEL)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    ax2.set_title(r"Cumulative number of profiles", fontsize=FONT_SIZE_TITLE)

    fig.tight_layout(pad=1.5)

    out_path = Path(__file__).parent.parent / "plots" / "figure-8.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigure saved to {out_path}")
