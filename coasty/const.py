r"""Global constants shared across the codebase."""

# fmt: off
#
# Determine hypoxia based on oxygen concentration threshold [µmol/kg].
HYPOXIA_THRESHOLD = 63.0

# Define bin size for spatial aggregation of profiles into sites [km]
BIN_SIZE = 5

# Approximate km-to-degree conversion for lat/lon binning (1° ≈ 111 km)
KM_PER_DEG = 111.0
