"""Sphinx configuration for the ``pyDeepLearn`` API documentation.

Build the HTML documentation from the repository root with::

    uv run --group docs sphinx-build -b html docs docs/_build/html

or simply ``make docs``.
"""

import os
import sys

# Make the package importable when the docs are built from a source checkout.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "pyDeepLearn"
copyright = "2022, Cory W. Mauer"
author = "Cory W. Mauer"

# These two values are rewritten in place by python-semantic-release (see the
# `version_variables` setting in pyproject.toml) whenever a release is cut, so
# they must stay valid SemVer strings with no interpolation.
release = "1.0.1"
version = "1.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The project uses NumPy-style docstrings, so enable Napoleon's NumPy parsing
# and disable the Google-style parser.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
