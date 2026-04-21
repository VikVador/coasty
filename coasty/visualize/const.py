r"""Global constants shared across the visualization modules."""

import cmocean as cmo

# fmt: off
#
# --- Font sizes ---
FONT_SIZE_TITLE      = 16         # Main plot title
FONT_SIZE_SUBTITLE   = 14         # Subtitle or panel label (e.g. "(a)", "(b)")
FONT_SIZE_X_LABEL    = 13         # X-axis label
FONT_SIZE_Y_LABEL    = 13         # Y-axis label
FONT_SIZE_TICK       = 11         # Tick labels on both axes
FONT_SIZE_LEGEND     = 11         # Legend text
FONT_SIZE_COLORBAR   = 11         # Colorbar tick labels and label
FONT_SIZE_ANNOTATION = 10         # In-plot text annotations

# --- Figure sizes (width, height) in inches ---
FIGURE_SIZE_SQUARE = (8, 8)       # Equal axes, e.g. maps or scatter plots
FIGURE_SIZE_WIDE   = (12, 5)      # Wider than tall, e.g. time series
FIGURE_SIZE_TALL   = (6, 10)      # Taller than wide, e.g. vertical profiles

# --- DPI ---
FIGURE_DPI      = 100             # Screen display
FIGURE_DPI_SAVE = 300             # Saved figures (publications, reports)

# --- Line and marker styles ---
LINE_WIDTH        = 1.5           # Default line width
LINE_WIDTH_THIN   = 0.8           # Secondary or background lines
LINE_WIDTH_THICK  = 2.5           # Highlighted or foreground lines
MARKER_SIZE       = 6             # Default marker size
MARKER_SIZE_SMALL = 3             # Dense scatter or secondary data
MARKER_SIZE_LARGE = 10            # Highlighted points

# --- Padding and spacing ---
TIGHT_LAYOUT_PAD    = 1.5         # Padding around subplots (plt.tight_layout)
COLORBAR_PAD        = 0.02        # Gap between plot and colorbar
COLORBAR_FRACTION   = 0.046       # Colorbar width relative to the axes
LEGEND_FRAMEALPHA   = 0.8         # Legend background opacity

# --- Alpha values ---
ALPHA_FILL    = 0.3               # Shaded regions (e.g. uncertainty bands)
ALPHA_GRID    = 0.4               # Grid lines
ALPHA_SCATTER = 0.7               # Scatter plot points

# --- Colormaps ---
CMAP_OXYGEN_SEQUENTIAL = cmo.haline   # Sequential colormap for oxygen concentration
CMAP_OXYGEN_DIVERGING  = cmo.balance  # Diverging colormap for oxygen anomalies
