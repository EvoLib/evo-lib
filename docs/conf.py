# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EvoLib"
copyright = "2025, EvoLib"
author = "EvoLib"
release = "0.1.0adev3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = "alabaster"
#html_theme = "sphinx_rtd_theme"
html_theme = "furo"
html_static_path = ["_static"]


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
    "sphinx_autodoc_typehints",  # Typannotationen automatisch darstellen
    "sphinx.ext.viewcode",       # Verlinkung zu Quellcode
    "sphinx.ext.intersphinx",    # Optionale Querverlinkung (z. B. NumPy, Python-Stdlib)
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
