import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE
from coasty.visualize.const import CMAP_SAMPLING_COUNT, FIGURE_DPI_SAVE
from coasty.visualize.utils import add_size_legend, compute_site_stats, plot_hypoxia_map

# PROMPT
figure_prompt = """

    Show the spatial sampling effort (total number of profiles per 10 km site) worldwide,
    for the full 1950–present period and for each decade separately.  Dot size and color
    both encode the profile count (log scale).  A size legend shows three reference circles
    representing < 10, < 100, and < 1000 profiles per site.

"""

# Configurable circle-size thresholds for the legend
CIRCLE_THRESHOLDS = [10, 100, 1000]

if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values

    # Dummy hypoxic flag — only the "total" key from compute_site_stats is used here
    is_hypoxic_dummy = np.zeros(len(lat), dtype=bool)

    decades = list(range(1950, 2030, 10))
    time_periods = [("all", None)] + [(str(d), d) for d in decades]

    out_dir = Path(__file__).parent.parent / "plots" / "figure-7"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_fig = None

    for period_name, decade in time_periods:
        if decade is None:
            mask = np.ones(len(lat), dtype=bool)
            title_suffix = r"$1950$--present"
        else:
            mask = (years >= decade) & (years < decade + 10)
            title_suffix = rf"${decade}$s"

        if mask.sum() == 0:
            continue

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic_dummy[mask], bin_size=BIN_SIZE)
        total = stats["total"].astype(float)

        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{len(total):>6,} sites | max count={total.max():.0f}"
        )

        color_vmax = 1000 if decade is None else 100
        fig, ax = plot_hypoxia_map(
            site_lats=stats["lat"],
            site_lons=stats["lon"],
            metric=total,
            title=rf"Sampling effort -- {title_suffix}",
            cbar_label=r"Number of profiles",
            cmap=CMAP_SAMPLING_COUNT,
            log_scale=True,
            vmax=color_vmax,
        )
        add_size_legend(ax, CIRCLE_THRESHOLDS, total, log_scale=True)

        fig.savefig(
            out_dir / f"figure-7-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
