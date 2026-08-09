r"""Figure 1: Global map of hypoxic percentage per site, by decade."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE
from coasty.visualize.const import FIGURE_DPI_SAVE
from coasty.visualize.hypoxia_maps import (
    make_category_map,
    new_map_axes,
)
from coasty.visualize.utils import (
    compute_hypoxic_flag,
    compute_site_stats,
)


if __name__ == "__main__":
    #
    # Set math font to Computer Modern for LaTeX-style labels
    plt.rcParams["mathtext.fontset"] = "cm"

    # Loading data
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    is_hypoxic = compute_hypoxic_flag(ds)

    print(f"  Profiles: {len(lat):,}  |  Hypoxic: {is_hypoxic.sum():,}")

    # Time periods to plot: "all" + individual decades
    decades = list(range(1950, 2030, 10))
    time_periods = [("all", None)] + [(str(d), d) for d in decades]

    out_dir = Path(__file__).parent.parent / "plots" / "figure-1"
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

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask], bin_size=BIN_SIZE)
        show = stats["n_hypoxic"] > 0

        if show.sum() == 0:
            continue

        s_lat = stats["lat"]#[show]
        s_lon = stats["lon"]#[show]
        pct = stats["pct_hypoxic"]#[show]
        n_hypoxic = stats["n_hypoxic"]#[show]

        mean_pct = stats["pct_hypoxic"].mean()
        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{len(stats['lat']):>5,} sites | {show.sum():>4,} hypoxic | "
            f"mean pct={mean_pct:.1f}%"
        )

        fig, ax = new_map_axes()
        make_category_map(
            ax,
            s_lat=s_lat,
            s_lon=s_lon,
            pct=pct,
            n_hypoxic=n_hypoxic,
            title=rf"Hypoxic percentage -- {title_suffix}",
        )
        fig.subplots_adjust(right=0.78)
        fig.savefig(
            out_dir / f"figure-1-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
