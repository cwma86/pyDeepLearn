# pyDeepLearn developer tasks
#
# This project uses uv (https://docs.astral.sh/uv/) to manage the Python
# version, the virtual environment, and all dependencies. There is no
# requirements.txt; dependencies live in pyproject.toml and are locked in
# uv.lock.

SHELL := /bin/bash

# Build a wheel and sdist into dist/
pkg:
	@uv build

# Create/refresh the virtual environment and install all dependencies
# (runtime plus the "dev" and "docs" dependency groups)
env:
	@uv sync --all-groups

# Run the unit test suite
check:
	@uv run python -m unittest discover -s pyDeepLearn

# Lint with flake8
lint:
	@uv run flake8 .

# Build the Sphinx API documentation into docs/_build/html
docs:
	@uv run --group docs sphinx-build -b html docs docs/_build/html

# Preview the next semantic release without changing any files, commits, or tags
release-check:
	@uv run --group release semantic-release -v --noop version

# Remove build artifacts, the virtual environment, and caches
clean:
	@rm -rf build dist *.egg-info .venv docs/_build \
		./pyDeepLearn/__pycache__ ./pyDeepLearn/tests/__pycache__
