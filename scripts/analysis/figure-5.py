r"""Figure 5: AOU (Apparent Oxygen Utilization).

This script computes and visualizes the Apparent Oxygen Utilization (AOU) for
each profile in the coastal dataset at its bottom depth.

AOU is defined as: AOU = O2_solubility - O2_observed

This represents the oxygen deficit, i.e., how much oxygen has been consumed
relative to the saturation value. Positive AOU indicates undersaturation,
negative AOU indicates supersaturation.

Generates:
  1. A single global spatial map for all time periods combined
  2. Decadal maps for each decade

Reference: See section 2.3 of WOA23 Oxygen documentation
https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DOCUMENTATION/WOA23_Oxygen.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET_SURFACE
from coasty.const import BIN_SIZE
from coasty.diagnosis import compute_aou
from coasty.visualize.const import CMAP_OXYGEN_DIVERGING, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_site_median, plot_hypoxia_map

# PROMPT
figure_prompt = """
    Create maps showing the mean AOU [µmol/kg] for the coastal zone. First,
    display a single global map for all time periods combined. Then, create
    decadal maps. AOU is computed as O2_solubility - O2_observed (DOX2).
    Aggregate profiles into BIN_SIZE-km sites and compute the median AOU for
    each site. Display as maps with the diverging oxygen colormap to highlight
    positive (undersaturated) and negative (supersaturated) values.
    
    Reference: WOA23 Oxygen documentation section 2.3
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
    print("Computing AOU...")
    aou_mean = compute_aou(ds, depth="bottom")
    
    valid = np.isfinite(aou_mean)
    print(f"  Profiles with valid AOU: {valid.sum():,} / {len(aou_mean):,}")
    
    out_dir = Path(__file__).parent.parent.parent / "plots" / "analysis" / "figure-5"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    last_fig = None
    
    # --- First: Spatial map for all time combined ---
    print("\nAggregating all data spatially...")
    site_stats = compute_site_median(
        lat[valid], 
        lon[valid], 
        aou_mean[valid], 
        bin_size=BIN_SIZE
    )
    valid_sites = np.isfinite(site_stats["median"])
    
    print(f"  Total sites: {valid_sites.sum():,}")
    
    # Use a reasonable vmax based on typical AOU range
    median_aou = site_stats["median"][valid_sites]
    vmax = max(abs(median_aou.min()), abs(median_aou.max()), 100.0)
    
    fig, ax = plot_hypoxia_map(
        site_lats=site_stats["lat"][valid_sites],
        site_lons=site_stats["lon"][valid_sites],
        metric=median_aou,
        title=r"Median AOU ($O_2^{{solubility}} - O_2$) -- All time",
        cbar_label=r"AOU $[\mu\mathrm{mol\,kg}^{-1}]$",
        cmap=CMAP_OXYGEN_DIVERGING,
        scale_marker_size=False,
        vmax=vmax,
    )
    
    fig.savefig(
        out_dir / "figure-5-aou-all-space.pdf",
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
            aou_mean[mask], 
            bin_size=BIN_SIZE
        )
        valid_sites = np.isfinite(site_stats["median"])
        
        print(
            f"  {decade}s: {mask.sum():,} profiles | "
            f"{valid_sites.sum():,} sites"
        )
        
        # Create map visualization
        # For AOU with diverging colormap, we use plot_hypoxia_map
        # The vmax parameter controls the upper bound for color normalization
        median_aou = site_stats["median"][valid_sites]
        # Use a reasonable vmax based on the data range
        vmax = max(abs(median_aou.min()), abs(median_aou.max()), 100.0)
        
        fig, ax = plot_hypoxia_map(
            site_lats=site_stats["lat"][valid_sites],
            site_lons=site_stats["lon"][valid_sites],
            metric=median_aou,
            title=rf"Median AOU ($O_2^{{solubility}} - O_2$) -- ${decade}$s",
            cbar_label=r"AOU $[\mu\mathrm{mol\,kg}^{-1}]$",
            cmap=CMAP_OXYGEN_DIVERGING,
            scale_marker_size=False,
            vmax=vmax,
        )
        
        fig.savefig(
            out_dir / f"figure-5-aou-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig
    
    if last_fig is not None:
        plt.show()
    
    print(f"\nFigures saved to {out_dir}/") 