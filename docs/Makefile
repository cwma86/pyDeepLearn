# Minimal makefile for Sphinx documentation
#
# The canonical entry point for this project is the ``docs`` target in the
# repository root makefile, which uses ``uv`` to run Sphinx:
#
#     make docs
#
# This file exists so that ``docs/`` can also be built on its own.

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	@rm -rf "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
