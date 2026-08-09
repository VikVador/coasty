r"""Figure 1: Vertically averaged oxygen inventory of the global coastal zone.

This script computes and visualizes the vertically averaged oxygen inventory
for each profile in the coastal dataset, aggregated by site.

The vertically averaged oxygen inventory is the mean DOX2 concentration over
the water column for each profile.

Generates:
  1. A single global spatial map for all time periods combined
  2. Decadal maps for each decade
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE, HYPOXIA_THRESHOLD
from coasty.diagnosis import compute_vertically_averaged_oxygen_inventory
from coasty.visualize.const import CMAP_OXYGEN_SEQUENTIAL, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_site_median, plot_hypoxia_map

# PROMPT
figure_prompt = """
    Create maps showing the vertically averaged oxygen inventory [µmol/kg]
    for the coastal zone. First, display a single global map for all time
    periods combined. Then, create decadal maps. Aggregate profiles into 
    BIN_SIZE-km sites and compute the median vertically averaged oxygen
    for each site. Display as maps with the sequential oxygen colormap.
"""


if __name__ == "__main__":
    plt.rcParams["mathtext.fontset"] = "cm"
    
    # --- Load data ---
    print("Loading dataset...")
    ds = xr.open_dataset(PATH_DATASET, engine="netcdf4")
    
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    years = ds["obs_time"].dt.year.values
    
    print(f"  Profiles: {len(lat):,}")
    
    # --- Compute diagnostic ---
    print("Computing vertically averaged oxygen inventory...")
    vert_avg_o2 = compute_vertically_averaged_oxygen_inventory(ds)
    
    valid = np.isfinite(vert_avg_o2)
    print(f"  Profiles with valid vertically averaged O2: {valid.sum():,} / {len(vert_avg_o2):,}")
    
    out_dir = Path(__file__).parent.parent.parent / "plots" / "analysis" / "figure-1"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    last_fig = None
    
    # --- First: Spatial map for all time combined ---
    print("\nAggregating all data spatially...")
    site_stats = compute_site_median(
        lat[valid], 
        lon[valid], 
        vert_avg_o2[valid], 
        bin_size=BIN_SIZE
    )
    valid_sites = np.isfinite(site_stats["median"])
    
    n_hypoxic = (site_stats["median"][valid_sites] < HYPOXIA_THRESHOLD).sum()
    print(f"  Total sites: {valid_sites.sum():,} | Hypoxic sites: {n_hypoxic}")
    
    fig, ax = plot_hypoxia_map(
        site_lats=site_stats["lat"][valid_sites],
        site_lons=site_stats["lon"][valid_sites],
        metric=site_stats["median"][valid_sites],
        title=r"Median vertically averaged $O_2$ -- All time",
        cbar_label=r"Vertically averaged $O_2$ $[\mu\mathrm{mol\,kg}^{-1}]$",
        cmap=CMAP_OXYGEN_SEQUENTIAL,
        scale_marker_size=False,
    )
    
    fig.savefig(
        out_dir / "figure-1-vertically-averaged-o2-all-space.pdf",
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
            vert_avg_o2[mask], 
            bin_size=BIN_SIZE
        )
        valid_sites = np.isfinite(site_stats["median"])
        
        n_hypoxic = (site_stats["median"][valid_sites] < HYPOXIA_THRESHOLD).sum()
        print(
            f"  {decade}s: {mask.sum():,} profiles | "
            f"{valid_sites.sum():,} sites | {n_hypoxic} hypoxic sites"
        )
        
        # Create map visualization
        fig, ax = plot_hypoxia_map(
            site_lats=site_stats["lat"][valid_sites],
            site_lons=site_stats["lon"][valid_sites],
            metric=site_stats["median"][valid_sites],
            title=rf"Median vertically averaged $O_2$ -- ${decade}$s",
            cbar_label=r"Vertically averaged $O_2$ $[\mu\mathrm{mol\,kg}^{-1}]$",
            cmap=CMAP_OXYGEN_SEQUENTIAL,
            scale_marker_size=False,
        )
        
        fig.savefig(
            out_dir / f"figure-1-vertically-averaged-o2-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig
    
    if last_fig is not None:
        plt.show()
    
    print(f"\nFigures saved to {out_dir}/")