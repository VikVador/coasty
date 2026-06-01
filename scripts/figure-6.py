import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib.lines import Line2D
from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE, HYPOXIA_THRESHOLD
from coasty.visualize.const import (
    ALPHA_SCATTER,
    CMAP_OXYGEN_SEQUENTIAL,
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    FIGURE_DPI,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_MAP,
    FONT_SIZE_COLORBAR,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LEGEND_FRAMEALPHA,
    LINE_WIDTH_THIN,
    MARKER_SIZE_SMALL,
)
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

HYPOXIC_COLOR = "red"


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


def plot_oxygen_map(
    median_o2: np.ndarray,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    title: str,
) -> plt.Figure:
    r"""Create a world map of median deepest O2 with hypoxic sites highlighted in red.

    Non-hypoxic sites are colored by the sequential oxygen colormap; hypoxic sites
    (median O2 < HYPOXIA_THRESHOLD) are overplotted in red.  A dashed line on the
    colorbar marks the hypoxia threshold.

    Arguments:
        - median_o2 : Median deepest O2 per site [µmol/kg].
        - s_lat     : Site latitudes [°].
        - s_lon     : Site longitudes [°].
        - title     : Figure title (LaTeX mathtext).

    Returns:
        - fig : The created Figure.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)

    cmap_obj = plt.get_cmap(CMAP_OXYGEN_SEQUENTIAL)
    norm = mcolors.Normalize(vmin=np.nanmin(median_o2), vmax=np.nanmax(median_o2))

    hypoxic = median_o2 < HYPOXIA_THRESHOLD
    normal = ~hypoxic
    marker_s = MARKER_SIZE_SMALL**2

    # Normal sites — colored by oxygen concentration
    if normal.sum() > 0:
        ax.scatter(
            s_lon[normal],
            s_lat[normal],
            s=marker_s,
            c=median_o2[normal],
            cmap=cmap_obj,
            norm=norm,
            alpha=ALPHA_SCATTER,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    # Hypoxic sites — highlighted in red on top
    if hypoxic.sum() > 0:
        ax.scatter(
            s_lon[hypoxic],
            s_lat[hypoxic],
            s=marker_s,
            color=HYPOXIC_COLOR,
            alpha=ALPHA_SCATTER,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    # Colorbar (represents the full oxygen range)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
        shrink=0.8,
    )
    cbar.set_label(
        r"Median deepest $O_2$ $[\mu\mathrm{mol\,kg}^{-1}]$",
        fontsize=FONT_SIZE_COLORBAR,
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Mark the hypoxia threshold on the colorbar
    cbar.ax.axhline(
        y=norm(HYPOXIA_THRESHOLD),
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    cbar.ax.text(
        1.15,
        norm(HYPOXIA_THRESHOLD),
        rf"${HYPOXIA_THRESHOLD:.0f}$",
        va="center",
        ha="left",
        fontsize=FONT_SIZE_TICK - 1,
        transform=cbar.ax.transAxes,
    )

    # Legend entry for hypoxic sites
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=MARKER_SIZE_SMALL,
                markerfacecolor=HYPOXIC_COLOR,
                markeredgecolor="none",
                label=rf"Hypoxic ($< {HYPOXIA_THRESHOLD:.0f}\ \mu\mathrm{{mol\,kg}}^{{-1}}$)",
            )
        ],
        fontsize=FONT_SIZE_LEGEND,
        loc="lower left",
        framealpha=LEGEND_FRAMEALPHA,
    )

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    fig.tight_layout(pad=1.5)
    return fig


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
