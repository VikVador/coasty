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


def compute_o2_saturation_garcia_gordon(temp_c: np.ndarray, sal_psu: np.ndarray) -> np.ndarray:
    r"""Compute oxygen saturation concentration using Garcia & Gordon (1992).
    
    This computes the dissolved oxygen concentration at 100% saturation
    (in µmol/kg) for given temperature and salinity.
    
    Arguments:
        - temp_c : Temperature in degrees Celsius [°C]. Can be array-like.
        - sal_psu : Practical Salinity [PSU]. Can be array-like.
    
    Returns:
        - o2_sat : Oxygen saturation concentration [µmol/kg].
    
    Notes:
        The Garcia & Gordon (1992) formula computes solubility in ml/l at STP.
        We convert to µmol/kg using:
            1 ml/l O2 = 44.66 µmol/l ≈ 44.66 µmol/kg (assuming density ≈ 1 kg/l)
    """
    temp_c = np.asarray(temp_c, dtype=float)
    sal_psu = np.asarray(sal_psu, dtype=float)
    
    # Convert to Kelvin
    temp_k = temp_c + 273.15
    
    # Avoid division by zero and log of zero/negative
    valid = (temp_k > 0) & (sal_psu >= 0)
    
    o2_sat = np.full_like(temp_c, np.nan, dtype=float)
    
    if not np.any(valid):
        return o2_sat
    
    T = temp_k[valid]
    S = sal_psu[valid]
    
    # Compute ln(O2_solubility) in ml/l at 1 atm
    ln_c = (
        A1 
        + A2 * 100.0 / T 
        + A3 * np.log(T / 100.0) 
        + A4 * T / 100.0 
        + S * (B1 + B2 * T / 100.0 + B3 * (T / 100.0) ** 2)
    )
    
    # Convert from ln(ml/l) to ml/l
    c_ml_per_l = np.exp(ln_c)
    
    # Convert from ml/l to µmol/kg
    # 1 ml O2 at STP = 1/22.4 mol = 1000/22.4 µmol = 44.642857 µmol
    # So 1 ml/l = 44.642857 µmol/l ≈ 44.642857 µmol/kg (density ≈ 1 kg/l)
    o2_sat_molal = c_ml_per_l * 44.642857
    
    o2_sat[valid] = o2_sat_molal
    
    return o2_sat


def compute_o2_saturation_umol_per_kg(temp_c: np.ndarray, sal_psu: np.ndarray) -> np.ndarray:
    r"""Wrapper for oxygen saturation calculation with explicit unit in name.
    
    Same as compute_o2_saturation_garcia_gordon but with more explicit naming.
    
    Arguments:
        - temp_c : Temperature in degrees Celsius [°C].
        - sal_psu : Practical Salinity [PSU].
    
    Returns:
        - o2_sat : Oxygen saturation concentration [µmol/kg].
    """
    return compute_o2_saturation_garcia_gordon(temp_c, sal_psu)


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


def compute_o2_saturation(
    ds,
) -> np.ndarray:
    r"""Compute O2 saturation concentration for each observation.
    
    This computes the oxygen concentration that would be present at 100% saturation
    for the given temperature and salinity conditions.
    
    Arguments:
        - ds : Loaded coasty dataset with TEMP and PSAL.
    
    Returns:
        - o2_sat : Array of same shape as observations with O2 saturation [µmol/kg].
    """
    temp = ds["TEMP"].values.astype(float)
    sal = ds["PSAL"].values.astype(float)
    
    return compute_o2_saturation_garcia_gordon(temp, sal)


def compute_solubility_percentage(
    ds,
    o2_sat: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute solubility percentage (O2_measure / O2_saturation * 100).
    
    This represents the observed oxygen concentration as a percentage of the
    saturation concentration.
    
    Arguments:
        - ds : Loaded coasty dataset with DOX2, TEMP, PSAL.
        - o2_sat : Optional pre-computed O2 saturation [µmol/kg]. If None, will be computed.
    
    Returns:
        - solubility_pct : Array of same shape as observations with solubility percentage [%].
    """
    dox2 = ds["DOX2"].values.astype(float)
    
    if o2_sat is None:
        o2_sat = compute_o2_saturation(ds)
    
    # Avoid division by zero
    valid = (o2_sat > 0) & np.isfinite(dox2) & np.isfinite(o2_sat)
    
    solubility_pct = np.full_like(dox2, np.nan, dtype=float)
    solubility_pct[valid] = 100.0 * dox2[valid] / o2_sat[valid]
    
    return solubility_pct


def compute_aou(
    ds,
    o2_sat: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compute Apparent Oxygen Utilization (AOU = O2_saturation - O2_observed).
    
    AOU represents the oxygen deficit, i.e., how much oxygen has been consumed
    relative to the saturation value. Positive AOU indicates undersaturation,
    negative AOU indicates supersaturation.
    
    Arguments:
        - ds : Loaded coasty dataset with DOX2, TEMP, PSAL.
        - o2_sat : Optional pre-computed O2 saturation [µmol/kg]. If None, will be computed.
    
    Returns:
        - aou : Array of same shape as observations with AOU [µmol/kg].
    """
    dox2 = ds["DOX2"].values.astype(float)
    
    if o2_sat is None:
        o2_sat = compute_o2_saturation(ds)
    
    valid = np.isfinite(dox2) & np.isfinite(o2_sat)
    
    aou = np.full_like(dox2, np.nan, dtype=float)
    aou[valid] = o2_sat[valid] - dox2[valid]
    
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


def compute_per_profile_o2_saturation_mean(ds) -> np.ndarray:
    r"""Compute mean O2 saturation for each profile.
    
    Arguments:
        - ds : Loaded coasty dataset.
    
    Returns:
        - profile_o2_sat_mean : Array of shape (n_profiles,) with mean O2 saturation [µmol/kg].
    """
    o2_sat = compute_o2_saturation(ds)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)
    
    n_profiles = len(starts)
    profile_o2_sat_mean = np.full(n_profiles, np.nan, dtype=float)
    
    for i in range(n_profiles):
        profile_o2_sat = o2_sat[starts[i]:ends[i]]
        valid = np.isfinite(profile_o2_sat)
        if np.sum(valid) > 0:
            profile_o2_sat_mean[i] = np.mean(profile_o2_sat[valid])
    
    return profile_o2_sat_mean


def compute_per_profile_solubility_pct_mean(ds) -> np.ndarray:
    r"""Compute mean solubility percentage for each profile.
    
    Arguments:
        - ds : Loaded coasty dataset.
    
    Returns:
        - profile_solubility_pct_mean : Array of shape (n_profiles,) with mean solubility [%].
    """
    solubility_pct = compute_solubility_percentage(ds)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)
    
    n_profiles = len(starts)
    profile_solubility_pct_mean = np.full(n_profiles, np.nan, dtype=float)
    
    for i in range(n_profiles):
        profile_solubility = solubility_pct[starts[i]:ends[i]]
        valid = np.isfinite(profile_solubility)
        if np.sum(valid) > 0:
            profile_solubility_pct_mean[i] = np.mean(profile_solubility[valid])
    
    return profile_solubility_pct_mean


def compute_per_profile_aou_mean(ds) -> np.ndarray:
    r"""Compute mean AOU for each profile.
    
    Arguments:
        - ds : Loaded coasty dataset.
    
    Returns:
        - profile_aou_mean : Array of shape (n_profiles,) with mean AOU [µmol/kg].
    """
    aou = compute_aou(ds)
    starts = ds["profile_start"].values.astype(int) - 1
    ends = ds["profile_end"].values.astype(int)
    
    n_profiles = len(starts)
    profile_aou_mean = np.full(n_profiles, np.nan, dtype=float)
    
    for i in range(n_profiles):
        profile_aou = aou[starts[i]:ends[i]]
        valid = np.isfinite(profile_aou)
        if np.sum(valid) > 0:
            profile_aou_mean[i] = np.mean(profile_aou[valid])
    
    return profile_aou_mean
