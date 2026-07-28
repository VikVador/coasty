r"""Oxygen diagnostic functions for the Coasty project.

This module provides functions to compute various oxygen-related diagnostics:
- Vertically averaged oxygen inventory
- Thickness of bottom hypoxic layer
- pO2 (partial pressure of O2, biological indicator)
- O2 saturation (solubility-based)
- Solubility percentage (O2_measure / O2_saturation)
- AOU (Apparent Oxygen Utilization)

References:
    - WOA23 Oxygen documentation: https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DOCUMENTATION/WOA23_Oxygen.pdf
    - Garcia, H.E., and Gordon, L.I., 1992: Oxygen solubility in seawater: Better fitting equations. Limnology and Oceanography, 37(6), 1307-1312.
"""

import gsw
import numpy as np

from coasty.const import HYPOXIA_THRESHOLD


# =============================================================================
# OXYGEN SOLUBILITY CALCULATION (Garcia & Gordon 1992)
# =============================================================================
# Coefficients for O2 solubility in seawater (ml/l)
# From Garcia & Gordon 1992, Table 1 (equation 8)
# Note: We use the coefficients for the natural log formulation

# For O2 solubility in ml/l at 1 atm, in terms of T (Kelvin) and S (PSU)
# ln(C) = A1 + A2*100/T + A3*ln(T/100) + A4*T/100 + S*(B1 + B2*T/100 + B3*(T/100)^2)

# Coefficients from Garcia & Gordon 1992 (in ln(ml/l) units)
A1 = -173.4292
A2 = 249.6339
A3 = 143.3483
A4 = -21.8492
B1 = -0.033096
B2 = 0.014259
B3 = -0.0017000


def compute_o2_solubility_garcia_gordon(
    temp_c: np.ndarray,
    sal_psu: np.ndarray,
    p_dbar: np.ndarray | float | None = None,
    lon_deg: np.ndarray | float | None = None,
    lat_deg: np.ndarray | float | None = None,
) -> np.ndarray:
    r"""Compute oxygen solubility with TEOS-10 GSW (gsw.O2sol).

    Arguments:
        - temp_c : In-situ temperature [degC].
        - sal_psu : Practical Salinity [PSU].
        - p_dbar : Sea pressure [dbar]. If None, uses 0 dbar.
        - lon_deg : Longitude [degE]. If None, uses 0.
        - lat_deg : Latitude [degN]. If None, uses 0.

    Returns:
        - o2_sol : Oxygen solubility [umol/kg].
    """
    temp_c = np.asarray(temp_c, dtype=float)
    sal_psu = np.asarray(sal_psu, dtype=float)

    # Build broadcastable auxiliary arrays for pressure and position.
    if p_dbar is None:
        p = np.zeros_like(temp_c, dtype=float)
    else:
        p = np.broadcast_to(np.asarray(p_dbar, dtype=float), temp_c.shape)

    if lon_deg is None:
        lon = np.zeros_like(temp_c, dtype=float)
    else:
        lon = np.broadcast_to(np.asarray(lon_deg, dtype=float), temp_c.shape)

    if lat_deg is None:
        lat = np.zeros_like(temp_c, dtype=float)
    else:
        lat = np.broadcast_to(np.asarray(lat_deg, dtype=float), temp_c.shape)

    o2_sol = np.full_like(temp_c, np.nan, dtype=float)
    valid = (
        np.isfinite(temp_c)
        & np.isfinite(sal_psu)
        & np.isfinite(p)
        & np.isfinite(lon)
        & np.isfinite(lat)
        & (sal_psu >= 0.0)
    )
    if not np.any(valid):
        return o2_sol

    sp = sal_psu[valid]
    t = temp_c[valid]
    p_valid = p[valid]
    lon_valid = lon[valid]
    lat_valid = lat[valid]

    sa = gsw.SA_from_SP(sp, p_valid, lon_valid, lat_valid)
    ct = gsw.CT_from_t(sa, t, p_valid)
    o2_sol[valid] = gsw.O2sol(sa, ct, p_valid, lon_valid, lat_valid)
    return o2_sol


def compute_o2_solubility_umol_per_kg(temp_c: np.ndarray, sal_psu: np.ndarray) -> np.ndarray:
    r"""Wrapper for oxygen solubility calculation with explicit unit in name.
    
    Same as compute_o2_solubility_garcia_gordon but with more explicit naming.
    
    Arguments:
        - temp_c : Temperature in degrees Celsius [°C].
        - sal_psu : Practical Salinity [PSU].
    
    Returns:
        - o2_sol : Oxygen solubility concentration [µmol/kg].
    """
    return compute_o2_solubility_garcia_gordon(temp_c, sal_psu)


def compute_o2_saturation_garcia_gordon(temp_c: np.ndarray, sal_psu: np.ndarray) -> np.ndarray:
    r"""Backward-compatible alias for oxygen saturation calculation.

    Uses the same Garcia & Gordon implementation as solubility.
    """
    return compute_o2_solubility_garcia_gordon(temp_c, sal_psu)


def compute_o2_saturation(ds) -> np.ndarray:
    r"""Compute O2 saturation concentration for each observation.

    Returns observation-level saturation/solubility from TEMP and PSAL.
    """
    temp = ds["TEMP"].values.astype(float)
    sal = ds["PSAL"].values.astype(float)
    return compute_o2_saturation_garcia_gordon(temp, sal)


# =============================================================================
# MAIN DIAGNOSTIC FUNCTIONS
# =============================================================================

def compute_vertically_averaged_oxygen_inventory(
    ds,
    max_depth: float | None = None,
) -> np.ndarray:
    r"""Compute vertically averaged oxygen inventory for each profile.
    
    The vertically averaged oxygen inventory is the mean DOX2 concentration
    over the water column for each profile. Optionally limited to a maximum depth.
    
    Arguments:
        - ds : Loaded coasty dataset with DOX2, obs_depths, profile_start, profile_end.
        - max_depth : Optional maximum depth [m] to consider. If None, uses all depths.
    
    Returns:
        - vert_avg_o2 : Array of shape (n_profiles,) with vertically averaged oxygen [µmol/kg].
                       NaN for profiles with no valid data.
    """
    dox2 = ds["DOX2"].values.astype(float)
    depths = ds["obs_depths"].values.astype(float)
    starts = ds["profile_start"].values.astype(int) - 1  # convert to 0-indexed
    ends = ds["profile_end"].values.astype(int)  # 1-based, exclusive
    
    n_profiles = len(starts)
    vert_avg_o2 = np.full(n_profiles, np.nan, dtype=float)
    
    for i in range(n_profiles):
        start_idx = starts[i]
        end_idx = ends[i]
        
        profile_dox2 = dox2[start_idx:end_idx]
        profile_depths = depths[start_idx:end_idx]
        
        # Apply depth filter if specified
        if max_depth is not None:
            valid = profile_depths <= max_depth
        else:
            valid = np.isfinite(profile_dox2) & np.isfinite(profile_depths)
        
        # Only consider finite oxygen values
        valid = valid & np.isfinite(profile_dox2)
        
        if np.sum(valid) > 0:
            vert_avg_o2[i] = np.mean(profile_dox2[valid])
    
    return vert_avg_o2


def compute_bottom_hypoxic_layer_thickness(
    ds,
    hypoxia_threshold: float = HYPOXIA_THRESHOLD,
) -> np.ndarray:
    r"""Compute thickness of the bottom hypoxic layer for each profile.
    
    Only considers profiles where the deepest observation is hypoxic. For such profiles,
    counts consecutive hypoxic observations working upward from the bottom. When the first
    non-hypoxic observation is encountered, linear interpolation is used to estimate how
    much of the gap between the last hypoxic and first non-hypoxic is actually hypoxic.
    
    Algorithm:
    1. Check if the deepest observation is hypoxic. If not, return 0.
    2. Work upward from the deepest observation, counting consecutive hypoxic observations.
    3. When a non-hypoxic observation is encountered:
       - Use linear interpolation between the last hypoxic and this non-hypoxic observation
       - Estimate the depth where O2 crosses the threshold
       - Add this partial thickness to the total
    
    Example: If profile has [hypoxic point (50 µmol/kg), non-hypoxic point (100 µmol/kg)] at [100m, 90m],
    interpolate to find where O2 = 63 µmol/kg, then include that partial distance.
    
    Arguments:
        - ds : Loaded coasty dataset.
        - hypoxia_threshold : Oxygen concentration threshold [µmol/kg] for hypoxia.
                              Default is HYPOXIA_THRESHOLD from config.
    
    Returns:
        - thickness : Array of shape (n_profiles,) with bottom hypoxic layer thickness [m].
                      Returns 0 for profiles where the deepest observation is not hypoxic.
    """
    dox2 = ds["DOX2"].values.astype(float)
    depths = ds["obs_depths"].values.astype(float)
    starts = ds["profile_start"].values.astype(int) - 1  # 0-indexed
    ends = ds["profile_end"].values.astype(int)  # 1-based, exclusive
    
    n_profiles = len(starts)
    thickness = np.zeros(n_profiles, dtype=float)

    for i in range(n_profiles):
        start_idx = starts[i]
        end_idx = ends[i]
        
        profile_dox2 = dox2[start_idx:end_idx]
        profile_depths = depths[start_idx:end_idx]
        
        # Remove NaN values
        valid_mask = np.isfinite(profile_dox2) & np.isfinite(profile_depths)
        if not np.any(valid_mask):
            thickness[i] = 0.0
            continue
        
        profile_dox2 = profile_dox2[valid_mask]
        profile_depths = profile_depths[valid_mask]
        
        # Sort by depth (just in case, though data should already be sorted)
        sort_idx = np.argsort(profile_depths)
        profile_dox2 = profile_dox2[sort_idx]
        profile_depths = profile_depths[sort_idx]
        
        # Check if the deepest observation is hypoxic
        if profile_dox2[-1] >= hypoxia_threshold:
            # Deepest point is not hypoxic, so no bottom hypoxic layer
            thickness[i] = 0.0
            continue
        
        # Work upward from the deepest observation
        # Find consecutive hypoxic observations from bottom
        n_obs = len(profile_dox2)

        deepest_depth = profile_depths[-1]
        boundary_depth = None  # Initialize boundary depth
        
        for j in range(n_obs - 2, -1, -1):  # Work upward from second-to-last
            if profile_dox2[j] < hypoxia_threshold:
                # Still hypoxic, continue upward
                continue
            else:
                # Hit a non-hypoxic observation, use interpolation at boundary
                # Linear interpolation between last hypoxic (at j+1) and first non-hypoxic (at j)
                idx_hypoxic = j + 1
                idx_non_hypoxic = j
                
                depth_hypoxic = profile_depths[idx_hypoxic]
                depth_non_hypoxic = profile_depths[idx_non_hypoxic]
                o2_hypoxic = profile_dox2[idx_hypoxic]
                o2_non_hypoxic = profile_dox2[idx_non_hypoxic]
                
                # Linear interpolation: find depth where O2 = threshold

                # depth = depth_hypoxic + (threshold - o2_hypoxic) / (o2_non_hypoxic - o2_hypoxic) * (depth_non_hypoxic - depth_hypoxic)
                boundary_depth = depth_hypoxic + (hypoxia_threshold - o2_hypoxic) / (o2_non_hypoxic - o2_hypoxic) * (depth_non_hypoxic - depth_hypoxic)

                
                # Stop searching upward
                break

        if boundary_depth is None:
            # All observations are hypoxic, so the boundary is at the shallowest observation
            boundary_depth = profile_depths[0]

        # Compute thickness from deepest to the boundary (including interpolated part)
        thickness[i] = deepest_depth - boundary_depth
    
    return thickness


def compute_po2(
    ds,
    o2_sat: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute partial pressure of O2 (pO2) for each observation.
    
    pO2 is a biological indicator that represents the partial pressure of oxygen.
    It can be computed as:
        pO2 = (DOX2 / O2_saturation) * atmospheric_pressure_fraction
    
    However, since we typically don't have atmospheric pressure, and for seawater
    at near-surface conditions, we can approximate pO2 as:
        pO2 = DOX2 / O2_saturation * 100
    
    This gives pO2 as a percentage of saturation.
    
    Alternatively, pO2 can be expressed in different units. Here we compute
    it as the ratio DOX2/O2_saturation (dimensionless), which is equivalent to
    the fraction of saturation.
    
    Arguments:
        - ds : Loaded coasty dataset with DOX2, TEMP, PSAL.
        - o2_sat : Optional pre-computed O2 saturation [µmol/kg]. If None, will be computed.
    
    Returns:
        - po2 : Array of same shape as DOX2 with pO2 values (dimensionless ratio).
    """
    dox2 = ds["DOX2"].values.astype(float)
    temp = ds["TEMP"].values.astype(float)
    sal = ds["PSAL"].values.astype(float)
    
    # Compute O2 saturation if not provided
    if o2_sat is None:
        o2_sat = compute_o2_saturation_garcia_gordon(temp, sal)
    
    # Avoid division by zero
    valid = (o2_sat > 0) & np.isfinite(dox2) & np.isfinite(o2_sat)
    
    po2 = np.full_like(dox2, np.nan, dtype=float)
    po2[valid] = dox2[valid] / o2_sat[valid]
    
    return po2


def compute_o2_solubility(
    ds,
    depth: str = "surface"
) -> np.ndarray:
    r"""Compute per-profile O2 solubility at surface or bottom.

    For each profile:
      - depth="surface": use the first measurement in the profile.
      - depth="bottom": use the last measurement in the profile.

    Arguments:
        - ds : Loaded coasty dataset with TEMP, PSAL, obs_depths,
               profile_start, and profile_end.
        - depth : Depth level to compute solubility for ("surface" or "bottom").

    Returns:
        - o2_sol : Array of shape (n_profiles,) with O2 solubility [µmol/kg].
    """
    if depth not in {"surface", "bottom"}:
        raise ValueError(f"Invalid depth argument: {depth}. Must be 'surface' or 'bottom'.")

    temp = ds["TEMP"].values.astype(float)
    sal = ds["PSAL"].values.astype(float)
    obs_depths = ds["obs_depths"].values.astype(float)
    lat = ds["latitude"].values.astype(float)
    lon = ds["longitude"].values.astype(float)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)

    n_profiles = len(starts)
    selected_temp = np.full(n_profiles, np.nan, dtype=float)
    selected_sal = np.full(n_profiles, np.nan, dtype=float)
    selected_p = np.full(n_profiles, np.nan, dtype=float)
    selected_lat = np.full(n_profiles, np.nan, dtype=float)
    selected_lon = np.full(n_profiles, np.nan, dtype=float)

    for i in range(n_profiles):
        s = starts[i]
        e = ends[i]

        prof_temp = temp[s:e]
        prof_sal = sal[s:e]
        prof_depth = obs_depths[s:e]

        if len(prof_temp) == 0:
            continue

        if depth == "surface":
            idx = 0
        else:
            idx = len(prof_temp) - 1

        t = prof_temp[idx]
        s_ = prof_sal[idx]
        d = prof_depth[idx]
        lat_i = lat[i]
        lon_i = lon[i]
        if np.isfinite(t) and np.isfinite(s_) and np.isfinite(d) and np.isfinite(lat_i) and np.isfinite(lon_i):
            selected_temp[i] = t
            selected_sal[i] = s_
            selected_lat[i] = lat_i
            selected_lon[i] = lon_i
            selected_p[i] = gsw.p_from_z(-d, lat_i)

    return compute_o2_solubility_garcia_gordon(
        selected_temp,
        selected_sal,
        p_dbar=selected_p,
        lon_deg=selected_lon,
        lat_deg=selected_lat,
    )


def compute_saturation_percentage(
    ds,
    depth: str = "surface",
) -> np.ndarray:
    r"""Compute saturation percentage (O2_measure / O2_saturation * 100).
    
    This represents the observed oxygen concentration as a percentage of the
    saturation concentration.
    
    Arguments:
        - ds : Loaded coasty dataset with DOX2, TEMP, PSAL.
        - depth : Depth level to compute saturation for ("surface" or "bottom").
    
    Returns:
        - saturation_pct : Array of shape (n_profiles,) with saturation percentage [%].
    """
    if depth not in {"surface", "bottom"}:
        raise ValueError(f"Invalid depth argument: {depth}. Must be 'surface' or 'bottom'.")

    dox2 = ds["DOX2"].values.astype(float)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)

    n_profiles = len(starts)
    selected_dox2 = np.full(n_profiles, np.nan, dtype=float)

    for i in range(n_profiles):
        s = starts[i]
        e = ends[i]
        prof_dox2 = dox2[s:e]
        if len(prof_dox2) == 0:
            continue

        if depth == "surface":
            idx = 0
        else:
            idx = len(prof_dox2) - 1

        val = prof_dox2[idx]
        if np.isfinite(val):
            selected_dox2[i] = val

    o2_sol = compute_o2_solubility(ds, depth=depth)

    # Avoid division by zero
    valid = (o2_sol > 0) & np.isfinite(selected_dox2) & np.isfinite(o2_sol)

    saturation_pct = np.full(n_profiles, np.nan, dtype=float)
    saturation_pct[valid] = 100.0 * selected_dox2[valid] / o2_sol[valid]

    return saturation_pct


def compute_aou(
    ds,
    depth: str = "bottom",
) -> np.ndarray:
    r"""Compute Apparent Oxygen Utilization (AOU = O2_solubility - O2_observed).
    
    AOU represents the oxygen deficit, i.e., how much oxygen has been consumed
    relative to the saturation value. Positive AOU indicates undersaturation,
    negative AOU indicates supersaturation.
    
    Arguments:
        - ds : Loaded coasty dataset with DOX2, TEMP, PSAL.
        - depth : Depth level to compute AOU for ("surface" or "bottom").
    
    Returns:
        - aou : Array of same shape as observations with AOU [µmol/kg].
    """
    dox2 = ds["DOX2"].values.astype(float)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)

    n_profiles = len(starts)
    selected_dox2 = np.full(n_profiles, np.nan, dtype=float)

    for i in range(n_profiles):
        s = starts[i]
        e = ends[i]
        prof_dox2 = dox2[s:e]
        if len(prof_dox2) == 0:
            continue

        if depth == "surface":
            idx = 0
        else:
            idx = len(prof_dox2) - 1

        val = prof_dox2[idx]
        if np.isfinite(val):
            selected_dox2[i] = val
    

    o2_sol = compute_o2_solubility(ds, depth=depth)
    
    valid = np.isfinite(selected_dox2) & np.isfinite(o2_sol)
    
    aou = np.full_like(selected_dox2, np.nan, dtype=float)
    aou[valid] = o2_sol[valid] - selected_dox2[valid]
    
    return aou


# =============================================================================
# PER-PROFILE AGGREGATION FUNCTIONS
# =============================================================================

def compute_per_profile_po2_mean(ds) -> np.ndarray:
    r"""Compute mean pO2 for each profile.
    
    Arguments:
        - ds : Loaded coasty dataset.
    
    Returns:
        - profile_po2_mean : Array of shape (n_profiles,) with mean pO2 per profile.
    """
    po2 = compute_po2(ds)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)
    
    n_profiles = len(starts)
    profile_po2_mean = np.full(n_profiles, np.nan, dtype=float)
    
    for i in range(n_profiles):
        profile_po2 = po2[starts[i]:ends[i]]
        valid = np.isfinite(profile_po2)
        if np.sum(valid) > 0:
            profile_po2_mean[i] = np.mean(profile_po2[valid])
    
    return profile_po2_mean

