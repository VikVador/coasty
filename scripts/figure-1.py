import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.visualize.const import CMAP_HYPOXIA_COUNT, CMAP_HYPOXIA_PCT, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_hypoxic_flag, compute_site_stats, plot_hypoxia_map

if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    is_hypoxic = compute_hypoxic_flag(ds)

    print(f"  Profiles: {len(lat):,}  |  Hypoxic: {is_hypoxic.sum():,}")

    # --- Time periods: whole period then each decade ---
    decades = list(range(1950, 2030, 10))
    time_periods = [("all", None)] + [(str(d), d) for d in decades]

    # --- Output directory ---
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

        stats = compute_site_stats(lat[mask], lon[mask], is_hypoxic[mask])
        show = stats["n_hypoxic"] > 0

        if show.sum() == 0:
            continue

        s_lat = stats["lat"][show]
        s_lon = stats["lon"][show]
        count = stats["n_hypoxic"][show].astype(float)
        pct = stats["pct_hypoxic"][show]

        n_total_sites = len(stats["lat"])
        mean_pct = pct.mean()
        median_pct = float(np.median(pct))
        print(
            f"  {period_name:>8s}: {mask.sum():>7,} profiles | "
            f"{n_total_sites:>6,} sites | {show.sum():>5,} hypoxic sites | "
            f"pct mean={mean_pct:.1f}% median={median_pct:.1f}%"
        )

        # Figure — dot size encodes number of hypoxic profiles
        fig, _ = plot_hypoxia_map(
            site_lats=s_lat,
            site_lons=s_lon,
            metric=count,
            title=rf"Hypoxic profiles -- {title_suffix}",
            cbar_label=r"Number of hypoxic profiles",
            cmap=CMAP_HYPOXIA_COUNT,
            log_scale=True,
        )
        fig.savefig(
            out_dir / f"figure-1-count-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Figure — dot size encodes hypoxic percentage
        fig, _ = plot_hypoxia_map(
            site_lats=s_lat,
            site_lons=s_lon,
            metric=pct,
            title=rf"Hypoxic percentage -- {title_suffix}",
            cbar_label=r"Hypoxic percentage $[\%]$",
            cmap=CMAP_HYPOXIA_PCT,
            log_scale=False,
        )
        fig.savefig(
            out_dir / f"figure-1-pct-{period_name}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
