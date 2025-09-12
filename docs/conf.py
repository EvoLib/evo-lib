import os
import pathlib
import sys
import tomllib

sys.path.insert(0, os.path.abspath(".."))

project = "EvoLib"
author = "EvoLib"
copyright = "2025, EvoLib"

# --- Load version from pyproject.toml ---
pyproject_file = pathlib.Path(__file__).parent.parent / "pyproject.toml"
with open(pyproject_file, "rb") as f:
    pyproject = tomllib.load(f)

release = pyproject["project"]["version"]

autodoc_typehints = "signature"
napoleon_attr_annotations = False
autodoc_member_order = "groupwise"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
    "member-order": "groupwise",
}

# ----------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}
