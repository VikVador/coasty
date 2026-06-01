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

# PROMPT
figure_prompt = """

    Show the annual sampling effort broken down by instrument type (Water bottle, Argo float,
    CTD) from 1950 to present.  Panel (a) shows stacked annual profile counts per instrument.
    Panel (b) shows the cumulative total number of profiles over time.
    Instrument types are now stored as string codes (BO, PF, CT) in the dataset.

"""

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
    for code in DATATYPE_LABELS:
        dt_mask = datatypes == code
        counts[code] = np.array([(years[dt_mask] == yr).sum() for yr in all_years])
        print(f"  datatype {code}: {dt_mask.sum():,} profiles total")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    bottom = np.zeros(len(all_years))
    for code in DATATYPE_LABELS:
        ax.bar(
            all_years,
            counts[code],
            bottom=bottom,
            color=DATATYPE_COLORS[code],
            label=DATATYPE_LABELS[code],
            alpha=0.85,
        )
        bottom += counts[code]

    ax.set_ylabel(r"Number of profiles", fontsize=FONT_SIZE_Y_LABEL)
    ax.set_xlabel(r"Year", fontsize=FONT_SIZE_X_LABEL)
    ax.set_title(
        r"Sampling effort -- annual profile count by instrument type ($1950$--present)",
        fontsize=FONT_SIZE_TITLE,
    )
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    fig.tight_layout(pad=1.5)

    out_path = Path(__file__).parent.parent / "plots" / "figure-8.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigure saved to {out_path}")
