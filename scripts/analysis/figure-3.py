r"""Figure 3: O2 solubility at the surface.

This script computes and visualizes the O2 solubility at the surface 
for each profile in the coastal dataset.

O2 solubility is computed using the Garcia & Gordon (1992) equation based on 
temperature and salinity. This represents the oxygen concentration that
would be present at 100% saturation for the given environmental conditions.

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
from coasty.diagnosis import compute_o2_solubility
from coasty.visualize.const import CMAP_OXYGEN_SEQUENTIAL, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_site_median, plot_hypoxia_map

# PROMPT
figure_prompt = """
    Create maps showing the mean O2 solubility [µmol/kg] for the
    coastal zone. First, display a single global map for all time periods combined.
    Then, create decadal maps. O2 solubility is computed using Garcia & Gordon
    (1992) solubility equation based on temperature and salinity. Aggregate
    profiles into BIN_SIZE-km sites and compute the median O2 solubility for
    each site. Display as maps with the sequential oxygen colormap.
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
    print("Computing O2 solubility...")
    o2_sol_mean = compute_o2_solubility(ds, depth="bottom")
    
    valid = np.isfinite(o2_sol_mean)
    print(f"  Profiles with valid O2 solubility: {valid.sum():,} / {len(o2_sol_mean):,}")
    
    out_dir = Path(__file__).parent.parent.parent / "plots" / "analysis" / "figure-3"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    last_fig = None
    
    # --- First: Spatial map for all time combined ---
    print("\nAggregating all data spatially...")
    site_stats = compute_site_median(
        lat[valid], 
        lon[valid], 
        o2_sol_mean[valid], 
        bin_size=BIN_SIZE
    )
    valid_sites = np.isfinite(site_stats["median"])
    
    print(f"  Total sites: {valid_sites.sum():,}")
    
    fig, ax = plot_hypoxia_map(
        site_lats=site_stats["lat"][valid_sites],
        site_lons=site_stats["lon"][valid_sites],
        metric=site_stats["median"][valid_sites],
        title=r"Median $O_2$ solubility -- All time",
        cbar_label=r"$O_2$ solubility $[\mu\mathrm{mol\,kg}^{-1}]$",
        cmap=CMAP_OXYGEN_SEQUENTIAL,
        scale_marker_size=False,
    )
    
    fig.savefig(
        out_dir / "figure-3-o2-solubility-bottom.pdf",
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
            o2_sol_mean[mask], 
            bin_size=BIN_SIZE
        )
        valid_sites = np.isfinite(site_stats["median"])
        
        print(
            f"  {decade}s: {mask.sum():,} profiles | "
            f"{valid_sites.sum():,} sites"
        )
        
        # Create map visualization
        fig, ax = plot_hypoxia_map(
            site_lats=site_stats["lat"][valid_sites],
            site_lons=site_stats["lon"][valid_sites],
            metric=site_stats["median"][valid_sites],
            title=rf"Median $O_2$ solubility -- ${decade}$s",
            cbar_label=r"$O_2$ solubility $[\mu\mathrm{mol\,kg}^{-1}]$",
            cmap=CMAP_OXYGEN_SEQUENTIAL,
            scale_marker_size=False,
        )
        
        fig.savefig(
            out_dir / f"figure-3-o2-solubility-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig
    
    if last_fig is not None:
        plt.show()
    
    print(f"\nFigures saved to {out_dir}/")