# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "pyperfectforesight"
author = "Shunsuke Hori"
release = "0.1.0"
copyright = "2026, Shunsuke Hori"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST configuration ------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/Shunsuke-Hori/pyperfectforesight",
}

html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

# autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# autosummary
autosummary_generate = True

# napoleon
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}
