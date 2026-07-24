import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE, MIN_CONSIDERED_YEAR, MAX_CONSIDERED_YEAR
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

    Show the annual time series of (a) the number of hypoxic sites and (b) the mean
    hypoxic percentage from 1950 to present.  Include a 10-year rolling mean on each panel.
    Mean hypoxic percentage is averaged over ALL sites (including non-hypoxic ones) to
    avoid inflation from single-profile sites.

"""

if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    is_hypoxic = compute_hypoxic_flag(ds)

    # --- Compute annual statistics ---
    all_years = np.arange(MIN_CONSIDERED_YEAR, MAX_CONSIDERED_YEAR + 1)
    n_hypoxic_sites = np.zeros(len(all_years), dtype=float)
    mean_pct = np.zeros(len(all_years), dtype=float)

    for i, yr in enumerate(all_years):
        mask = years == yr
        if mask.sum() == 0:
            n_hypoxic_sites[i] = np.nan
            mean_pct[i] = np.nan
            continue

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask], bin_size=BIN_SIZE)
        n_hypoxic_sites[i] = (stats["n_hypoxic"] > 0).sum()
        # Mean over ALL sites (zeros included) to avoid inflation from single-profile sites
        mean_pct[i] = stats["pct_hypoxic"].mean()

        print(
            f"  {yr}: {mask.sum():>6,} profiles | {n_hypoxic_sites[i]:>4.0f} hypoxic sites | mean pct={mean_pct[i]:.1f}%"
        )

    # --- 10-year rolling mean ---
    roll_sites = pd.Series(n_hypoxic_sites).rolling(10, min_periods=3, center=True).mean().values
    roll_pct = pd.Series(mean_pct).rolling(10, min_periods=3, center=True).mean().values

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE_WIDE, sharex=True)

    # Panel (a) — hypoxic site count
    ax1.bar(all_years, n_hypoxic_sites, color="steelblue", alpha=0.5, label=r"Annual count")
    ax1.plot(
        all_years,
        roll_sites,
        color="navy",
        linewidth=LINE_WIDTH_THICK,
        label=r"10-year rolling mean",
    )
    ax1.set_ylabel(r"Number of hypoxic sites", fontsize=FONT_SIZE_Y_LABEL)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    ax1.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left")
    ax1.set_title(
        rf"Annual hypoxic site count and mean hypoxic percentage (${MIN_CONSIDERED_YEAR}$--${MAX_CONSIDERED_YEAR}$)",
        fontsize=FONT_SIZE_TITLE,
    )

    # Panel (b) — mean hypoxic percentage
    ax2.bar(all_years, mean_pct, color="darkorange", alpha=0.5, label=r"Annual mean")
    ax2.plot(
        all_years,
        roll_pct,
        color="saddlebrown",
        linewidth=LINE_WIDTH_THICK,
        label=r"10-year rolling mean",
    )
    ax2.set_ylabel(r"Mean hypoxic percentage $[\%]$", fontsize=FONT_SIZE_Y_LABEL)
    ax2.set_xlabel(r"Year", fontsize=FONT_SIZE_X_LABEL)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    ax2.legend(fontsize=FONT_SIZE_LEGEND, loc="upper left")

    fig.tight_layout(pad=1.5)

    out_path = Path(__file__).parent.parent / "plots" / "figure-3.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")

    plt.show()
    print(f"\nFigure saved to {out_path}")
