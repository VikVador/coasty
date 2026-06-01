import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.lines import Line2D
from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import KM_PER_DEG
from coasty.visualize.const import (
    ALPHA_SCATTER,
    FIGURE_DPI,
    FIGURE_DPI_SAVE,
    FIGURE_SIZE_MAP,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LEGEND_FRAMEALPHA,
    LINE_WIDTH_THIN,
    MARKER_SIZE_SMALL,
    SEASON_COLORS,
)

# PROMPT
figure_prompt = """

    Show the seasonal coverage of ocean sampling aggregated by decade (1950s–2020s).
    For each site and decade, count how many of the four meteorological seasons
    (Spring, Summer, Autumn, Winter) have at least one observed profile.
    Color each site by that count (1–4) using four discrete colors.
    Repeat for two spatial aggregation radii: 5 km and 10 km.
    Produce one world map per (bin_km, decade) combination.

"""

# Spatial aggregation radii to study [km]
SPATIAL_BINS_KM = [200, 50]

# Season definitions: name → set of month numbers
SEASON_MONTHS = {
    "Spring": {4, 5, 6},
    "Summer": {7, 8, 9},
    "Autumn": {10, 11, 12},
    "Winter": {1, 2, 3},
}


def count_seasons_for_site(month_set: set) -> int:
    r"""Return the number of seasons that have at least one observation.

    Arguments:
        - month_set : Set of observed month numbers for a given site and decade.

    Returns:
        - n_seasons : Integer in {1, 2, 3, 4}.
    """
    return sum(1 for season_months in SEASON_MONTHS.values() if month_set & season_months)


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"

    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    months = ds["obs_time"].dt.month.values

    decades = list(range(1950, 2030, 10))
    out_dir = Path(__file__).parent.parent / "plots" / "figure-11"
    out_dir.mkdir(parents=True, exist_ok=True)

    last_fig = None

    for bin_km in SPATIAL_BINS_KM:
        bin_deg = bin_km / KM_PER_DEG

        # Compute site_id for every profile at this spatial resolution
        site_lat_arr = np.round(lat / bin_deg) * bin_deg
        site_lon_arr = np.round(lon / bin_deg) * bin_deg
        lat_idx = np.round((site_lat_arr + 90.0) / bin_deg).astype(int)
        lon_idx = np.round((site_lon_arr + 180.0) / bin_deg).astype(int)
        n_lon = int(np.ceil(360.0 / bin_deg)) + 1
        site_id = lat_idx * n_lon + lon_idx

        df = pd.DataFrame({"site_id": site_id, "year": years, "month": months})

        for decade in decades:
            decade_mask = (df["year"] >= decade) & (df["year"] < decade + 10)
            df_decade = df[decade_mask]

            if len(df_decade) == 0:
                continue

            # For each site in this decade, collect the set of observed months
            month_sets = df_decade.groupby("site_id")["month"].apply(set)

            # Count distinct seasons per site
            season_counts = month_sets.apply(count_seasons_for_site)

            # Reconstruct lat/lon for each unique site
            unique_site_ids = season_counts.index.values
            s_lat_idx = unique_site_ids // n_lon
            s_lon_idx = unique_site_ids % n_lon
            s_lat = s_lat_idx * bin_deg - 90.0
            s_lon = s_lon_idx * bin_deg - 180.0
            s_counts = season_counts.values

            print(
                f"  {bin_km} km, {decade}s: {len(s_lat):,} sites | "
                f"4-season: {(s_counts == 4).sum():,} ({100 * (s_counts == 4).mean():.1f}%)"
            )

            # --- Build Cartopy map ---
            fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
            ax.set_global()
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
            ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
            ax.add_feature(
                cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2
            )

            # One scatter call per season-count value (discrete colors)
            legend_handles = []
            for n_s in range(1, 5):
                sel = s_counts == n_s
                if sel.sum() == 0:
                    continue
                ax.scatter(
                    s_lon[sel],
                    s_lat[sel],
                    s=(MARKER_SIZE_SMALL * 2) ** 2,
                    color=SEASON_COLORS[n_s],
                    alpha=ALPHA_SCATTER,
                    linewidths=0,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="none",
                        markersize=MARKER_SIZE_SMALL * 2,
                        markerfacecolor=SEASON_COLORS[n_s],
                        markeredgecolor="none",
                        label=rf"${n_s}$ season{'s' if n_s > 1 else ''}",
                    )
                )

            ax.legend(
                handles=legend_handles,
                fontsize=FONT_SIZE_LEGEND,
                loc="lower left",
                framealpha=LEGEND_FRAMEALPHA,
            )
            ax.set_title(
                rf"Seasonal coverage -- ${bin_km}$ km -- ${decade}$s",
                fontsize=FONT_SIZE_TITLE,
            )
            ax.tick_params(labelsize=FONT_SIZE_TICK)
            fig.tight_layout(pad=1.5)

            fname = f"figure-11-{bin_km}km-{decade}.pdf"
            fig.savefig(out_dir / fname, dpi=FIGURE_DPI_SAVE, bbox_inches="tight")
            last_fig = fig

    if last_fig is not None:
        plt.show()

    print(f"\nFigures saved to {out_dir}/")
