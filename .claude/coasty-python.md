#
# Coding Guidelines
#

IMPORTANT: THIS FILE CONTAINS THE CODING GUIDELINES AND BEST PRACTICES FOR WORKING WITH THE COASTY DATASET AND CODEBASE.
           PLEASE REFER TO THIS DOCUMENT BEFORE WRITING ANY NEW CODE OR MAKING CHANGES TO EXISTING CODE. YOU SHOULD
           **STRICLY ADHERE TO THESE GUIDELINES** TO ENSURE CONSISTENCY, READABILITY, AND MAINTAINABILITY ACROSS THE PROJECT.
           DO NOT FORGET TO ACTIVATE THE `coasty` ENVIRONMENT BEFORE RUNNING ANY CODE, AS ALL NECESSARY DEPENDENCIES ARE INSTALLED
           THERE AND MAY NOT BE AVAILABLE IN OTHER ENVIRONMENTS. IF YOU ENCOUNTER MISSING DEPENDENCIES, CHECK THE `pyproject.toml`
           FILE AND INSTALL THE REQUIRED PACKAGES IN THE `coasty` ENVIRONMENT.

#
# -- General Philosophy --
#

The goal is to write Python code that is **clear, concise, and as simple as possible**.
Avoid over-engineering, prefer readable solutions over clever ones.
Always remember KISS: "Keep It Simple Stupid".

#
# -- Before writing code --
#

IMPORTANT: Before writing any code, check if what you want to implement
           already exists in the codebase (**Reuse existing functions whenever possible**).

#
# -- Rules --
#
- Keep functions **short and single-purpose**.
- Use **explicit variable names** that reflect the physical or mathematical meaning.
- Avoid unnecessary abstractions or complex class hierarchies.
- Prefer **built-in Python and NumPy/Xarray idioms** over custom implementations.
- Include **type hints** for function arguments and return values to improve readability and facilitate debugging.
- Include a **docstring** for every function following our personnal docstring format:

```python
def compute_hypoxic_layer(depths: np.ndarray, dox2: np.ndarray, bathymetry: float) -> float:
    r"""Compute the thickness of the bottom hypoxic layer for a single profile.

    Arguments:
        - depths     : Observation depths for the profile, sorted in ascending order [m].
        - dox2       : Dissolved oxygen values for the profile [µmol/kg].
        - bathymetry : Seafloor depth at the profile location [m].

    Returns:
        - thickness : Thickness of the bottom hypoxic layer [m]. Returns 0 if no hypoxia is detected.
    """
```
#
# -- Figure Scripts --
#

IMPORTANT: Each time you work on a figure's script
- Make sure that the script **always follow** the guidelines above.
- Make sure that the script **follows the same structure as other scripts** :

```python

# PROMPT


figure_prompt = """

    Our prompt to generate the figure will be written by ourselves here.

"""


# INCLUDE ALL THE LIBRARIES YOU NEED TO GENERATE THE FIGURE HERE

# FUNCTIONS
# DEFINE ALL THE FUNCTIONS YOU NEED IN ADDITION TO THE ONES ALREADY IN THE CODEBASE HERE (ONLY IF THEY DO NOT EXIST IN THE CODEBASE)


if __name__ == "__main__":

    #
    # THIS IS WHERE YOU WRITE THE CODE TO GENERATE THE FIGURE, USING THE ONES IN THE CODEBASE AND THE ONES YOU DEFINED ABOVE
    #
```

#
# -- Notebook --
#
- In `notebook/coasty.ipynb`, we will execute in each cell a single figure script from the `scripts/` directory.
- After you finish a figure script, make sure to add a cell in the notebook to execute it.
- Each figure script should display the figure in the notebook but also save it in the `plots/` directory in .pdf format.
