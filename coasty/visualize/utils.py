r"""Visualization and analysis utility functions for the Coasty project."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from typing import cast

from coasty.const import (
    BIN_SIZE,
    HYPOXIA_THRESHOLD,
    KM_PER_DEG,
)

from .const import (
    ALPHA_SCATTER,
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    FIGURE_DPI,
    FIGURE_SIZE_MAP,
    FONT_SIZE_COLORBAR,
    FONT_SIZE_LEGEND,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    LEGEND_FRAMEALPHA,
    LINE_WIDTH_THIN,
    MARKER_SIZE_LARGE,
    MARKER_SIZE_SMALL,
)


def compute_hypoxic_flag(ds: xr.Dataset) -> np.ndarray:
    r"""Return a boolean array indicating whether each profile is hypoxic.

    A profile is flagged hypoxic if at least one observation has DOX2 < 63 µmol/kg.

    Arguments:
        - ds : Loaded coasty dataset with variables DOX2 and profile_start.

    Returns:
        - is_hypoxic : Boolean array of shape (n_profiles,).
    """
    dox2 = ds["DOX2"].values.astype(float)
    np.nan_to_num(dox2, nan=np.inf, copy=False)  # NaN → inf so they never trigger hypoxia

    profile_start = ds["profile_start"].values.astype(int) - 1  # 0-indexed

    min_dox2 = np.minimum.reduceat(dox2, profile_start)

    return min_dox2 < HYPOXIA_THRESHOLD


def compute_site_stats(
    lat: np.ndarray,
    lon: np.ndarray,
    is_hypoxic: np.ndarray,
    bin_size: float = BIN_SIZE,
) -> dict:
    r"""Aggregate profiles into ~5 km sites and compute hypoxia statistics.

    Arguments:
        - lat        : Profile latitudes [°].
        - lon        : Profile longitudes [°].
        - is_hypoxic : Boolean hypoxic flag per profile.
        - bin_size   : Binning resolution [km]. 
    Returns:
        - stats : Dict with keys 'lat', 'lon', 'total', 'n_hypoxic', 'pct_hypoxic'.
    """
    bin_deg = bin_size / KM_PER_DEG

    site_lat = np.round(lat / bin_deg) * bin_deg
    site_lon = np.round(lon / bin_deg) * bin_deg

    lat_idx = np.round((site_lat + 90.0) / bin_deg).astype(int)
    lon_idx = np.round((site_lon + 180.0) / bin_deg).astype(int)

    n_lon = int(np.ceil(360.0 / bin_deg)) + 1  # safe encoding factor
    site_id = lat_idx * n_lon + lon_idx

    unique_ids, inverse = np.unique(site_id, return_inverse=True)

    total = np.bincount(inverse)
    n_hypoxic = np.bincount(inverse, weights=is_hypoxic.astype(float)).astype(int)
    pct = 100.0 * n_hypoxic / np.maximum(total, 1)

    out_lat = (unique_ids // n_lon) * bin_deg - 90.0
    out_lon = (unique_ids % n_lon) * bin_deg - 180.0

    return {
        "lat": out_lat,
        "lon": out_lon,
        "total": total,
        "n_hypoxic": n_hypoxic,
        "pct_hypoxic": pct,
    }


def compute_site_median(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    bin_size: float = BIN_SIZE,
) -> dict:
    r"""Aggregate profiles into sites and return the per-site median of a variable.

    NaN values in `values` are ignored when computing the median.

    Arguments:
        - lat      : Profile latitudes [°].
        - lon      : Profile longitudes [°].
        - values   : Per-profile scalar values (e.g. bottom oxygen).
        - bin_size : Binning resolution [km]. Default BIN_SIZE.

    Returns:
        - result : Dict with keys 'lat', 'lon', 'median'.
    """
    bin_deg = bin_size / KM_PER_DEG

    site_lat = np.round(lat / bin_deg) * bin_deg
    site_lon = np.round(lon / bin_deg) * bin_deg

    lat_idx = np.round((site_lat + 90.0) / bin_deg).astype(int)
    lon_idx = np.round((site_lon + 180.0) / bin_deg).astype(int)

    n_lon = int(np.ceil(360.0 / bin_deg)) + 1
    site_id = lat_idx * n_lon + lon_idx

    unique_ids, inverse = np.unique(site_id, return_inverse=True)

    medians = np.array([np.nanmedian(values[inverse == k]) for k in range(len(unique_ids))])

    out_lat = (unique_ids // n_lon) * bin_deg - 90.0
    out_lon = (unique_ids % n_lon) * bin_deg - 180.0

    return {"lat": out_lat, "lon": out_lon, "median": medians}


def compute_bottom_oxygen(ds: xr.Dataset) -> np.ndarray:
    r"""Return per-profile mean DOX2 in the bottom 15 m layer.

    The bottom layer is defined as depths in [bathymetry - 15, bathymetry].
    Profiles with no observations in this layer return NaN.

    Arguments:
        - ds : Loaded coasty dataset.

    Returns:
        - bottom_o2 : Array of shape (n_profiles,) [µmol/kg], NaN where unavailable.
    """
    dox2 = ds["DOX2"].values.astype(float)
    depths = ds["obs_depths"].values
    bathy = ds["bathymetry"].values
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)

    lengths = ends - starts
    obs_bathy = np.repeat(bathy, lengths)  # broadcast bathymetry to obs level

    valid = (depths >= obs_bathy - 15) & (depths <= obs_bathy) & np.isfinite(dox2)

    dox2_sum = np.add.reduceat(np.where(valid, dox2, 0.0), starts)
    count = np.add.reduceat(valid.astype(float), starts)

    count_safe = np.where(count > 0, count, 1)  # avoid div-by-zero in np.where
    return np.where(count > 0, dox2_sum / count_safe, np.nan)


def add_size_legend(
    ax: GeoAxes,
    thresholds: list[float],
    metric: np.ndarray,
    log_scale: bool = True,
    color: str = "gray",
    loc: str = "lower left",
) -> None:
    r"""Add a size reference legend to map axes using proxy Line2D markers.

    Proxy marker sizes are computed with the same formula as plot_hypoxia_map so
    that the legend accurately reflects the scatter-plot encoding.

    Arguments:
        - ax          : Cartopy GeoAxes or standard Axes to annotate.
        - thresholds  : Reference values shown in the legend (e.g. [10, 100, 1000]).
        - metric      : Full metric array used in the corresponding scatter plot.
        - log_scale   : True if plot_hypoxia_map was called with log_scale=True.
        - color       : Marker fill color for all proxy handles.
        - loc         : Legend location string (e.g. "lower left").
    """
    vals = np.log1p(metric.astype(float)) if log_scale else metric.astype(float)
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin + 1e-10

    handles = []
    for t in thresholds:
        val = np.log1p(float(t)) if log_scale else float(t)
        norm_val = (val - vmin) / span
        marker_r = MARKER_SIZE_SMALL + norm_val * (MARKER_SIZE_LARGE - MARKER_SIZE_SMALL)
        handle = Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=max(marker_r, 2.0),
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=rf"$< {int(t)}$",
        )
        handles.append(handle)

    ax.legend(handles=handles, fontsize=FONT_SIZE_LEGEND, loc=loc, framealpha=LEGEND_FRAMEALPHA)


def plot_hypoxia_map(
    site_lats: np.ndarray,
    site_lons: np.ndarray,
    metric: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str | Colormap,
    log_scale: bool = False,
    vmax: float | None = None,
    scale_marker_size: bool = True,
) -> tuple[Figure, GeoAxes]:
    r"""Create a global map with sites sized and colored by a scalar metric.

    Arguments:
        - site_lats  : Latitudes of sites [°].
        - site_lons  : Longitudes of sites [°].
        - metric     : Values driving dot size and color (e.g. count or percentage).
        - title      : Figure title (supports LaTeX mathtext).
        - cbar_label : Colorbar label (supports LaTeX mathtext).
        - cmap       : Matplotlib colormap instance.
        - log_scale  : Apply log normalisation to dot sizes. Default False.
        - vmax       : Upper bound for colormap and size normalisation. Values above
                       this are clipped to the maximum size/color. Default None (data max).
        - scale_marker_size : If True, marker size encodes metric magnitude.
                      If False, all markers use a fixed size.

    Returns:
        - fig, ax : Figure and Axes objects.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = cast(GeoAxes, fig.add_subplot(1, 1, 1, projection=ccrs.Robinson()))
    ax.set_global()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.add_feature(cfeature.OCEAN, color="white", zorder=0)
    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)

    if scale_marker_size:
        vals = np.log1p(metric) if log_scale else metric.copy()
        vmin_val = vals.min()
        vmax_val = (
            np.log1p(vmax)
            if (log_scale and vmax is not None)
            else (vmax if vmax is not None else vals.max())
        )
        vals_clipped = np.clip(vals, vmin_val, vmax_val)
        norm_vals = (vals_clipped - vmin_val) / (vmax_val - vmin_val + 1e-10)
        sizes = (MARKER_SIZE_SMALL + norm_vals * (MARKER_SIZE_LARGE - MARKER_SIZE_SMALL)) ** 2
    else:
        sizes = np.full(metric.shape, MARKER_SIZE_SMALL**2, dtype=float)

    color_vmax = vmax if vmax is not None else metric.max()
    sc = ax.scatter(
        site_lons,
        site_lats,
        s=sizes,
        c=metric,
        cmap=cmap,
        vmin=metric.min(),
        vmax=color_vmax,
        alpha=ALPHA_SCATTER,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="vertical",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
        shrink=0.8,
    )
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    fig.tight_layout(pad=1.5)

    return fig, ax


def plot_heatmap_2d(
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    values_2d: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str | Colormap,
) -> tuple[Figure, GeoAxes]:
    r"""Create a global map with a regular lat/lon grid colored by 2-D values.

    Arguments:
        - lat_centers : 1-D array of cell centre latitudes [°].
        - lon_centers : 1-D array of cell centre longitudes [°].
        - values_2d   : 2-D array of shape (n_lat, n_lon); NaN cells are transparent.
        - title       : Figure title (supports LaTeX mathtext).
        - cbar_label  : Colorbar label (supports LaTeX mathtext).
        - cmap        : Matplotlib colormap.

    Returns:
        - fig, ax : Figure and Axes objects.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_MAP, dpi=FIGURE_DPI)
    ax = cast(GeoAxes, fig.add_subplot(1, 1, 1, projection=ccrs.Robinson()))
    ax.set_global()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.add_feature(cfeature.LAND, color="#e8e8e8", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=LINE_WIDTH_THIN, edgecolor="0.4", zorder=2)

    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

    masked = np.ma.masked_invalid(values_2d)

    pm = ax.pcolormesh(
        lon2d,
        lat2d,
        masked,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        zorder=0,
    )

    cbar = fig.colorbar(
        pm,
        ax=ax,
        orientation="vertical",
        pad=COLORBAR_PAD,
        fraction=COLORBAR_FRACTION,
        shrink=0.8,
    )
    cbar.set_label(cbar_label, fontsize=FONT_SIZE_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    fig.tight_layout(pad=1.5)

    return fig, ax
