# Dataset Documentation

## Overview

This dataset contains vertical ocean profiles collected worldwide since 1950.
It is structured around two core dimensions: **profiles** (`time`) and **observations** (`values`).

## Loading the Dataset

```python
import xarray as xr
ds = xr.open_dataset(input_path)
```

## Structure

```
Dimensions:   (time: 457960, values: 11318578)
Coordinates:
  datatype    (time) int32        # Instrument type per profile
Data variables:
  obs_time      (time) datetime64   # Collection timestamp
  latitude      (time) float32      # Profile latitude
  longitude     (time) float32      # Profile longitude
  profile_start (time) int32        # Start index in 'values' arrays (1-indexed!)
  profile_end   (time) int32        # End index in 'values' arrays (1-indexed!)
  bathymetry    (time) float32      # Seafloor depth at profile location
  DOX2          (values) float32    # Dissolved oxygen
  PSAL          (values) float32    # Salinity
  TEMP          (values) float32    # Temperature
  obs_depths    (values) float32    # Observation depths
```

## Dimensions Explained

| Dimension | Size        | Represents                           |
|-----------|-------------|--------------------------------------|
| `time`    | ~457,960    | One entry per profile                |
| `values`  | ~11,318,578 | One entry per individual observation |

## Variables Explained

### Profile-level variables — dim: `time`

- **`datatype`**: Instrument used to collect the profile
  - `1` → Water bottle
  - `2` → Argo float
  - `3` → CTD sensor
- **`obs_time`**: Date and time the profile was collected
- **`latitude`**, **`longitude`**: Geographic coordinates of the profile
- **`bathymetry`**: Seafloor depth at the profile's location
- **`profile_start`** / **`profile_end`**: Indices into the `values`-dimension arrays marking where each profile begins and ends

  > ⚠️ **Important:** Indices are **1-based** (not 0-based). Adjust accordingly when slicing in Python.

### Observation-level variables — dim: `values`

All observations across all profiles are stored in flat arrays. Use `profile_start` and `profile_end` to extract individual profiles.

- **`DOX2`**: Dissolved oxygen measurements
- **`PSAL`**: Salinity measurements
- **`TEMP`**: Temperature measurements
- **`obs_depths`**: Depth of each observation

  > Within a profile, depths are **sorted in ascending order**, so the deepest observation is always at index `profile_end`.

## Extracting a Single Profile

```python
# Profile index i (0-based in Python)
start = int(ds["profile_start"][i]) - 1  # Convert to 0-based
end   = int(ds["profile_end"][i])        # Slice end is exclusive in Python

temp   = ds["TEMP"].values[start:end]
depths = ds["obs_depths"].values[start:end]
```
