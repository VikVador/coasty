r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
LOCAL          = Path("/gpfs/projects/acad/bsmfc/coasty/")
FOLDER_DATASET = LOCAL / "data"

# --- Datasets ---
#
# Path to our coastal oxygen dataset.
PATH_DATASET = FOLDER_DATASET / "unified_300_coastal_bottom15_with_bathymetry.nc"
