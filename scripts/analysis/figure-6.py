r"""Figure 6: Saturation percentage (O2_measure / O2_solubility * 100).

This script computes and visualizes the saturation percentage for each profile
in the coastal dataset.

Saturation percentage is computed as (DOX2 / O2_solubility) * 100, representing
the observed oxygen concentration as a percentage of the solubility concentration.
This is a direct measure of how saturated the water is with oxygen.

Generates:
  1. A single global spatial map for all time periods combined
  2. Decadal maps for each decade
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET_SURFACE
from coasty.const import BIN_SIZE
from coasty.diagnosis import compute_saturation_percentage
from coasty.visualize.const import CMAP_OXYGEN_DIVERGING, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_site_median, plot_hypoxia_map

PLOT_VMIN = 20.0
PLOT_VMAX = 150.0

# PROMPT
figure_prompt = """
    Create maps showing the mean saturation percentage [%] for the coastal zone.
    First, display a single global map for all time periods combined. Then, create
    decadal maps. Saturation percentage is (DOX2 / O2_solubility) * 100. Aggregate
    profiles into BIN_SIZE-km sites and compute the median saturation percentage
    for each site. Display as maps with the diverging oxygen colormap to highlight
    undersaturated (low %) and supersaturated (high %) regions.
"""


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"
    
    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET_SURFACE, engine="netcdf4")
    
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    
    print(f"  Profiles: {len(lat):,}")
    
    # --- Compute diagnostic ---
    print("Computing saturation percentage...")
    saturation_pct = compute_saturation_percentage(ds, depth="surface")
    
    valid = np.isfinite(saturation_pct)
    print(f"  Profiles with valid saturation %: {valid.sum():,} / {len(saturation_pct):,}")
    
    out_dir = Path(__file__).parent.parent.parent / "plots" / "analysis" / "figure-6"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    last_fig = None
    
    # --- First: Spatial map for all time combined ---
    print("\nAggregating all data spatially...")
    site_stats = compute_site_median(
        lat[valid], 
        lon[valid], 
        saturation_pct[valid], 
        bin_size=BIN_SIZE
    )
    valid_sites = np.isfinite(site_stats["median"])
    
    print(f"  Total sites: {valid_sites.sum():,}")

    metric_plot = np.clip(site_stats["median"][valid_sites], PLOT_VMIN, PLOT_VMAX)
    
    fig, ax = plot_hypoxia_map(
        site_lats=site_stats["lat"][valid_sites],
        site_lons=site_stats["lon"][valid_sites],
        metric=metric_plot,
        title=r"Median $O_2$ saturation percentage -- All time",
        cbar_label=r"$O_2$ saturation $[\%]$",
        cmap=CMAP_OXYGEN_DIVERGING,
        scale_marker_size=False,
        vmax=PLOT_VMAX,
    )
    
    fig.savefig(
        out_dir / "figure-6-saturation-percentage-surface.pdf",
        dpi=FIGURE_DPI_SAVE,
        bbox_inches="tight",
    )
    last_fig = fig
    
    # --- Then: Decadal maps ---
    print("\nGenerating decadal maps...")
    decades = list(range(1950, 2030, 10))
    
    for decade in decades:
        mask = (years >= decade) & (years < decade + 10) & valid
        
        if mask.sum() == 0:
            print(f"  {decade}s: no data, skipping")
            continue
        
        # Compute site statistics
        site_stats = compute_site_median(
            lat[mask], 
            lon[mask], 
            saturation_pct[mask], 
            bin_size=BIN_SIZE
        )
        valid_sites = np.isfinite(site_stats["median"])
        
        print(
            f"  {decade}s: {mask.sum():,} profiles | "
            f"{valid_sites.sum():,} sites"
        )

        metric_plot = np.clip(site_stats["median"][valid_sites], PLOT_VMIN, PLOT_VMAX)
        
        # Create map visualization
        fig, ax = plot_hypoxia_map(
            site_lats=site_stats["lat"][valid_sites],
            site_lons=site_stats["lon"][valid_sites],
            metric=metric_plot,
            title=rf"Median $O_2$ saturation percentage -- ${decade}$s",
            cbar_label=r"$O_2$ saturation $[\%]$",
            cmap=CMAP_OXYGEN_DIVERGING,
            scale_marker_size=False,
            vmax=PLOT_VMAX,
        )
        
        fig.savefig(
            out_dir / f"figure-6-saturation-percentage-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig
    
    if last_fig is not None:
        plt.show()
    
    print(f"\nFigures saved to {out_dir}/")