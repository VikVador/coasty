import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import CMAP_SAMPLING_COUNT, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_site_stats, plot_hypoxia_map

if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values

    # Dummy hypoxic flag — only "total" key is used for this sampling figure
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

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic_dummy[mask])
        total = stats["total"].astype(float)

        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{len(total):>6,} sites | max count={total.max():.0f}"
        )

        fig, _ = plot_hypoxia_map(
            site_lats=stats["lat"],
            site_lons=stats["lon"],
            metric=total,
            title=rf"Sampling effort -- {title_suffix}",
            cbar_label=r"Number of profiles",
            cmap=CMAP_SAMPLING_COUNT,
            log_scale=True,
        )

        suffix = "all" if period_name == "all" else period_name
        fig.savefig(
            out_dir / f"figure-7-{suffix}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
