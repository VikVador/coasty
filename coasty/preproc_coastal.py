import xarray as xr
import numpy as np

from coasty.config import PATH_DATASET, PATH_FULL_PROFILE_DATASET
from data.mast_structure.transform_mast_data import *


def filter_by_time(ds, start_year=1950):
    # Remove profiles collected before 1950 by building their profile indices
    profile_years = ds["obs_time"].dt.year.values
    to_remove = np.where(profile_years < start_year)[0]

    print(f"Total profiles: {len(profile_years)}")
    print(f"Profiles to remove (< {start_year}): {len(to_remove)}")
    print(f"Profiles kept (> {start_year}): {len(profile_years) - len(to_remove)}")

    ds = remove_profiles_by_index(ds, to_remove)

    return ds


def filter_profile_depth(ds, max_depth=200):
    # Apply an additional filter AFTER the year filter: remove profiles whose
    # depth at profile_end is > 200 m

    all_ends = ds["profile_end"].values
    obs_depths = ds["obs_depths"].values

    to_remove_depth = []

    for i, end_idx_1based in enumerate(all_ends):
        obs_idx = int(end_idx_1based) - 1
        d = obs_depths[obs_idx]
        if d > max_depth:
            to_remove_depth.append(i)

    to_remove_depth = np.array(to_remove_depth, dtype=int)

    print(f"Profiles to remove by profile_end depth > {max_depth}: {len(to_remove_depth)}")

    ds = remove_profiles_by_index(ds, to_remove_depth)

    return ds


def filter_bathymetry_depth(ds, max_bathymetry=200):
    # Find profiles over deep ocean (seafloor deeper than 200m, i.e., bathymetry < -200)
    # and remove them to retain only coastal/shallow profiles
    bathymetry_values = ds["bathymetry"].values
    to_remove_bathy = np.where(bathymetry_values < -max_bathymetry)[0]

    print(f"Total profiles: {len(bathymetry_values)}")
    print(f"Profiles to remove (bathymetry < -{max_bathymetry}m): {len(to_remove_bathy)}")
    print(f"Profiles kept (seafloor ≤ {max_bathymetry}m): {len(bathymetry_values) - len(to_remove_bathy)}")

    ds = remove_profiles_by_index(ds, to_remove_bathy)

    return ds


def remove_below_bathymetry_profiles(ds, max_excess=50):
    # Remove profiles whose deepest observation exceeds the local bathymetry by more than max_excess metres.
    # excess = max_observed_depth - abs(bathymetry); remove if excess > max_excess.

    all_ends = ds["profile_end"].values.astype(int) - 1  # convert to 0-indexed
    obs_depths = ds["obs_depths"].values
    bathymetry = ds["bathymetry"].values

    max_obs_depth = obs_depths[all_ends]
    excess = max_obs_depth - np.abs(bathymetry)
    to_remove = np.where(excess > max_excess)[0]

    print(f"Total profiles: {len(all_ends)}")
    print(f"Profiles to remove (deepest obs > bathymetry + {max_excess} m): {len(to_remove)}")
    print(f"Profiles kept: {len(all_ends) - len(to_remove)}")

    ds = remove_profiles_by_index(ds, to_remove)

    return ds


def remove_non_bottom_profiles(ds, max_diff = 15):
    # On the coastal subset, compare the deepest observed measurement with the local bathymetry.
    # depth_gap = abs(bathymetry) - max_observed_depth
    # Remove profile if depth_gap > 15 m; flag it for review if depth_gap < 0.

    bathymetry_values_coastal = ds["bathymetry"].values
    all_ends_coastal = ds["profile_end"].values
    obs_depths_coastal = ds["obs_depths"].values

    profiles_to_remove_gap = []
    depth_gaps = np.empty(len(all_ends_coastal), dtype=np.float32)

    for i, end_idx_1based in enumerate(all_ends_coastal):
        obs_idx = int(end_idx_1based) - 1
        max_observed_depth = obs_depths_coastal[obs_idx]
        max_depth = abs(bathymetry_values_coastal[i])
        depth_gap = max_depth - max_observed_depth
        depth_gaps[i] = depth_gap

        if depth_gap > max_diff:
            profiles_to_remove_gap.append(i)

    profiles_to_remove_gap = np.array(profiles_to_remove_gap, dtype=int)

    print(f"Total coastal profiles checked: {len(all_ends_coastal)}")
    print(f"Profiles to remove (bathymetry - max observed depth > {max_diff} m): {len(profiles_to_remove_gap)}")

    ds = remove_profiles_by_index(ds, profiles_to_remove_gap)

    return ds


def main():
    ds = xr.open_dataset(PATH_FULL_PROFILE_DATASET)
    ds = filter_by_time(ds, start_year=1950)
    ds = filter_profile_depth(ds, max_depth=200)
    ds = filter_bathymetry_depth(ds, max_bathymetry=200)
    ds = remove_below_bathymetry_profiles(ds, max_excess=50)
    ds = remove_non_bottom_profiles(ds, max_diff=15)
    ds.to_netcdf(PATH_DATASET)


if __name__ == "__main__":
    main()
