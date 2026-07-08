import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE, HYPOXIA_THRESHOLD
from coasty.visualize.const import FIGURE_DPI_SAVE

from coasty.visualize.hypoxia_maps import plot_oxygen_map
from coasty.visualize.utils import compute_site_median

# PROMPT
figure_prompt = """

    Show the spatial distribution of bottom-water oxygen across the globe, aggregated by
    decade (1950s–2020s).  For each profile, keep only the single deepest observation as a
    proxy for bottom oxygen.  Aggregate these deepest values per BIN_SIZE-km site using the
    median, and plot one world map per decade.  Sites whose median oxygen falls below the
    hypoxia threshold (HYPOXIA_THRESHOLD) are highlighted in red; all other sites use the
    sequential oxygen colormap so the colorbar remains informative.

"""


def extract_deepest_oxygen(ds: xr.Dataset) -> np.ndarray:
    r"""Return the dissolved oxygen value at the deepest observation of each profile.

    Depths within a profile are sorted in ascending order (shallowest first), so the
    deepest observation is always the last one, located at index profile_end - 1 (0-based).

    Arguments:
        - ds : Loaded coasty dataset.

    Returns:
        - deepest_o2 : Array of shape (n_profiles,) [µmol/kg].
    """
    dox2 = ds["DOX2"].values.astype(float)
    ends = ds["profile_end"].values.astype(int)  # 1-based → last obs at ends[i] - 1
    return dox2[ends - 1]


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    deepest_o2 = extract_deepest_oxygen(ds)

    valid_global = np.isfinite(deepest_o2)
    print(f"  Profiles with valid deepest O2: {valid_global.sum():,} / {len(deepest_o2):,}")

    decades = list(range(1950, 2030, 10))
    out_dir = Path(__file__).parent.parent / "plots" / "figure-6"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_fig = None

    for decade in decades:
        mask = (years >= decade) & (years < decade + 10) & valid_global

        if mask.sum() == 0:
            print(f"  {decade}s: no data, skipping")
            continue

        site_stats = compute_site_median(lat[mask], lon[mask], deepest_o2[mask], bin_size=BIN_SIZE)
        valid_sites = np.isfinite(site_stats["median"])

        n_hypoxic = (site_stats["median"][valid_sites] < HYPOXIA_THRESHOLD).sum()
        print(
            f"  {decade}s: {mask.sum():,} profiles | "
            f"{valid_sites.sum():,} sites | {n_hypoxic} hypoxic sites"
        )

        fig = plot_oxygen_map(
            median_o2=site_stats["median"][valid_sites],
            s_lat=site_stats["lat"][valid_sites],
            s_lon=site_stats["lon"][valid_sites],
            title=rf"Median deepest $O_2$ -- ${decade}$s",
        )
        fig.savefig(
            out_dir / f"figure-6-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
