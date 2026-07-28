r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
LOCAL          = Path("/gpfs/projects/acad/bsmfc/coasty/")
FOLDER_DATASET = LOCAL / "data"

GLOBAL_OCEAN_DATA = Path("/gpfs/projects/acad/bsmfc/cleaning_profiles")

# --- Datasets ---
#
# Path to our coastal oxygen datasets
PATH_FULL_PROFILE_DATASET = GLOBAL_OCEAN_DATA / "unified_1950_min3points.nc"
PATH_DATASET      = FOLDER_DATASET / "coastal200m_full.nc"
PATH_DATASET_SURFACE = FOLDER_DATASET / "surface2bottom.nc"
PATH_DATASET_DIAZ = FOLDER_DATASET / "dataset_oxygen_diaz_rosenberg.zarr"
