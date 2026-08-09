r"""Figure 2: Thickness of the bottom hypoxic layer.

This script computes and visualizes the thickness of the bottom hypoxic layer
for each profile in the coastal dataset.

The bottom hypoxic layer thickness is computed as the depth range where DOX2 <
HYPOXIA_THRESHOLD, measured from the deepest observation upward.

Generates:
  1. A single global spatial map for all time periods combined
  2. Decadal maps for each decade
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

from coasty.config import PATH_DATASET
from coasty.const import BIN_SIZE
from coasty.diagnosis import compute_bottom_hypoxic_layer_thickness
from coasty.visualize.const import CMAP_OXYGEN_DIVERGING, FIGURE_DPI_SAVE
from coasty.visualize.utils import compute_site_median, plot_hypoxia_map

# PROMPT
figure_prompt = """
    Create maps showing the thickness of the bottom hypoxic layer [m]
    for the coastal zone. First, display a single global map for all time
    periods combined. Then, create decadal maps. For each profile, compute
    the thickness as the depth range where DOX2 < HYPOXIA_THRESHOLD from
    the deepest observation upward. Aggregate profiles into BIN_SIZE-km
    sites and compute the median thickness for each site. Display as maps
    with the diverging oxygen colormap.
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
    print("Computing bottom hypoxic layer thickness...")
    hypoxic_thickness = compute_bottom_hypoxic_layer_thickness(ds)
    
    valid = hypoxic_thickness > 0  # Only consider profiles with hypoxic layer
    print(f"  Profiles with hypoxic layer: {valid.sum():,} / {len(hypoxic_thickness):,}")
    
    out_dir = Path(__file__).parent.parent.parent / "plots" / "analysis" / "figure-2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    last_fig = None
    
    # --- First: Spatial map for all time combined ---
    print("\nAggregating all data spatially...")
    site_stats = compute_site_median(
        lat[valid], 
        lon[valid], 
        hypoxic_thickness[valid], 
        bin_size=BIN_SIZE
    )
    valid_sites = np.isfinite(site_stats["median"])
    
    print(f"  Total sites with hypoxic layer: {valid_sites.sum():,}")
    
    fig, ax = plot_hypoxia_map(
        site_lats=site_stats["lat"][valid_sites],
        site_lons=site_stats["lon"][valid_sites],
        metric=site_stats["median"][valid_sites],
        title=r"Median bottom hypoxic layer thickness -- All time",
        cbar_label=r"Bottom hypoxic layer thickness $[\mathrm{m}]$",
        cmap=CMAP_OXYGEN_DIVERGING,
        scale_marker_size=False,
    )
    
    fig.savefig(
        out_dir / "figure-2-hypoxic-layer-thickness-all-space.pdf",
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
            hypoxic_thickness[mask], 
            bin_size=BIN_SIZE
        )
        valid_sites = np.isfinite(site_stats["median"])
        
        print(
            f"  {decade}s: {mask.sum():,} profiles | "
            f"{valid_sites.sum():,} sites with hypoxic layer"
        )
        
        # Create map visualization
        fig, ax = plot_hypoxia_map(
            site_lats=site_stats["lat"][valid_sites],
            site_lons=site_stats["lon"][valid_sites],
            metric=site_stats["median"][valid_sites],
            title=rf"Median bottom hypoxic layer thickness -- ${decade}$s",
            cbar_label=r"Bottom hypoxic layer thickness $[\mathrm{m}]$",
            cmap=CMAP_OXYGEN_DIVERGING,
            scale_marker_size=False,
        )
        
        fig.savefig(
            out_dir / f"figure-2-hypoxic-layer-thickness-{decade}.pdf",
            dpi=FIGURE_DPI_SAVE,
            bbox_inches="tight",
        )
        last_fig = fig
    
    if last_fig is not None:
        plt.show()
    
    print(f"\nFigures saved to {out_dir}/")