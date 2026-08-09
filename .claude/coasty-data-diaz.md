#
# Diaz Dataset
#

In this file, we provide an overview on how to access the Diaz hypoxia coastal zone dataset as well as its structure and key variables.

#
# -- Overview --
#

This dataset originates from **Diaz & Rosenberg (2008)** — *"Spreading Dead Zones and Consequences for Marine Ecosystems"*, Science, 321(5891), 926–929.
It catalogs **885 coastal hypoxic zones** identified worldwide, documenting for each site its geographic location, the decade of first observation, the type of hypoxia, and its current ecological classification. The dataset covers sites across **133 countries** and spans observations from the **1880s to the 2010s**. It is stored both as a raw Excel file (`hypoxia-dataset.xlsx`) and as a pre-processed Zarr archive (`hypoxia-dataset.zarr`).

#
# -- Loading the dataset --
#

IMPORTANT: DO NOT FORGET TO ACTIVATE THE `coasty` ENVIRONMENT BEFORE RUNNING THIS CODE,
           OTHERWISE YOU MAY ENCOUNTER ISSUES WITH MISSING DEPENDENCIES (e.g. zarr, xarray).
           ALL THE NECESSARY DEPENDENCIES ARE LISTED IN THE `pyproject.toml` AND ARE
           ALREADY INSTALLED WITHIN THE `coasty` ENVIRONMENT. IF A PACKAGE IS MISSING,
           ADD IT TO THE `pyproject.toml` FILE AND REINSTALL THE ENVIRONMENT.

```python
import xarray as xr
from coasty.config import PATH_DATASET_DIAZ

# Loading the dataset from path defined in the config file
ds = xr.open_zarr(PATH_DATASET_DIAZ)
```

#
# -- Structure of the dataset --
#

```
Dimensions: (profile: 885)
Coordinates:
  profile          (profile) int64     # Sequential index (0-based)
Data variables:
  lat              (profile) float32   # Latitude of the hypoxic zone
  lon              (profile) float32   # Longitude of the hypoxic zone
  decade           (profile) int32     # Decade of first hypoxia observation (-9999 = unknown)
  hypoxia_current  (profile) object    # Current hypoxia regime (see categories below)
  classification   (profile) object    # Current ecological status (see categories below)
```

The dimension can be explained as:

| Dimension | Size | Represents                        |
|-----------|------|-----------------------------------|
| `profile` | 885  | One entry per coastal hypoxic zone |

The variables can be defined as:

1. Geographic variables
  - `lat`, `lon`: Geographic coordinates (decimal degrees) of the hypoxic zone centroid.

2. Temporal variable
  - `decade`: Decade during which hypoxia was first reported at this site (e.g., `1990` = first observed in the 1990s).
    Missing values are encoded as `-9999`.

3. Hypoxia regime — `hypoxia_current`

  Describes how hypoxia manifests at the site:

  | Value        | Meaning                                                      |
  |--------------|--------------------------------------------------------------|
  | `Seasonal`   | Recurs annually, typically during warm months                |
  | `Episodic`   | Occurs irregularly, driven by weather or pulse nutrient loads|
  | `Persistent` | Permanent or near-permanent low-oxygen conditions            |
  | `Unknown`    | Regime not determined                                        |

  NOTE: The raw Excel file contains minor case inconsistencies (e.g., `'seasonal'` vs `'Seasonal'`).
        Normalize with `.str.strip().str.capitalize()` before analysis if needed.

4. Ecological classification — `classification`

  Reflects the current trophic or oxygen status of the zone:

  | Value       | Meaning                                                          |
  |-------------|------------------------------------------------------------------|
  | `Hypoxic`   | Currently hypoxic (DO ≤ 2 mg/L)                                  |
  | `Eutrophic` | Nutrient-enriched but not yet hypoxic                            |
  | `Improved`  | Previously hypoxic, now recovering due to management actions     |

  NOTE: Same case inconsistencies as `hypoxia_current` — normalize before analysis.

#
# -- Raw Excel columns (full dataset) --
#

The `hypoxia-dataset.xlsx` file contains additional columns not included in the Zarr extract:

| Column                 | Description                                           |
|------------------------|-------------------------------------------------------|
| `Country`              | Country or regional sea in which the site is located  |
| `Baltic Sea, Northern` | Site name / local geographic name                   |
| `Lat` / `Long`         | Decimal-degree coordinates                            |
| `Decade`               | Decade of first hypoxia observation                   |
| `Hypoxia Current`      | Current hypoxia regime (see above)                    |
| `Classification`       | Ecological status (see above)                         |
| `Benthic`              | Observed impacts on benthic (seafloor) communities    |
| `Fisheries`            | Observed impacts on fish and invertebrate populations |
| `Comment`              | Narrative description of the site and its history     |
| `References`           | Literature source(s) for the site record              |

NOTE: All column names in the raw file have **trailing whitespace** — strip with
      `df.columns = df.columns.str.strip()` immediately after loading.

#
# -- Reference --
#

Diaz, R. J., & Rosenberg, R. (2008). Spreading dead zones and consequences for marine ecosystems.
*Science*, 321(5891), 926–929. https://doi.org/10.1126/science.1156401
