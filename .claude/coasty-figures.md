#
# Figures General Guide
#

In this file, we will cover general guidelines for creating figures for the Coasty project.

#
# -- Global Constants --
#

IMPORTANT: All figures generated must use the same global constants for colors, fonts, and
           other styling elements to ensure a consistent look across all figures.

1. In `coasty/visualize/const.py`: You will find a set of global constants defined for figure styling, including font sizes, figure sizes, line widths, marker sizes, alpha values, and colormaps.

2. If a constant is missing that you think should be added, please add it to `const.py` and use it in your figure code **DO NOT ADD CONSTANTS IN THE FIGURE CODE FILES**.

#
# -- Figure Design Principles & Conventions --
#

1. **Simplicity**: When we ask you to show spatial distributions over some time period, for example showing results every decade, create **a figure per decade**.
                   So if possible, create multiple figures instead of trying to cram everything into one figure. This is because it is easier to read and interpret a single map.

2. **Clarity**: Use clear and descriptive titles, axis labels, and legends. Avoid cluttering the figure with too much information.

3. **Ocean Coastlines**: Do not forget to add ocean coastlines to your maps. This is important for context and readability.

4. **Reusable Code**: If you check that the code you need do not already exist, please add what you need to `coasty/visualize/utils.py` and reuse it in your figure code.

5. **Text**: Use Latex formatting for all text in the figures (titles, axis labels, legends, annotations) to ensure a professional and consistent appearance.

6. **Units**: Always include units in axis labels and legends where applicable. Also, use as format `[unit]`.

7. **Color Maps**: Use the colormaps defined in `const.py` for all figures. If you need a new colormap, add it to `const.py` and use it in your figure code.
                   You are allowed to use colormaps from the `matplotlib` and `cmocean` libraries, but make sure to define them in `const.py` and use the defined constants in your figure code.
                   (matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html, cmocean colormaps: https://matplotlib.org/cmocean/)

8. **Naming files**: Follow as naming convention for figure files: `figure-X.png` where X is the script number. If multiple figures are generated from the same script,
                     add a suffix to the file name: `figure-X-Y.png` where Y is a letter (a, b, c, etc.) indicating the figure number and place them in a subfolder
                     named `figure-X` (e.g., `figure-1/figure-1-a.png`, `figure-1/figure-1-b.png`, etc.).
