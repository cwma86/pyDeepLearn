# pyDeepLearn

[![Python application](https://github.com/cwma86/pyDeepLearn/actions/workflows/python-app.yml/badge.svg)](https://github.com/cwma86/pyDeepLearn/actions/workflows/python-app.yml)

A from-scratch, NumPy-based deep learning framework. `pyDeepLearn` was written
to explore the inner workings of neural network training -- forward and
backward propagation, layer composition, and objective (loss) functions --
without relying on an existing autograd or deep learning library.

## Features

- **Composable layers** built on a small `Layer` interface: input
  (z-score normalization), fully connected, recurrent, linear (identity), ReLU,
  sigmoid, tanh, and softmax.
- **Objective functions** built on a shared `objectiveFuncInterface`: least
  squares, log loss (binary cross entropy), and cross entropy.
- **Training helpers** for feed-forward networks (`run_layers`,
  `run_plot_epoch_J`) and recurrent networks (`RNN_train`, `RNN_predict`).
- **Pluggable weight updates**: plain gradient descent, recurrent updates, and
  the Adam optimizer.
- **Unit test suite** runnable with the standard library `unittest`.
- **Sphinx API documentation** generated from NumPy-style docstrings.

## Project structure

```
pyDeepLearn/
|-- main.py                  # command line demonstrations
|-- pyproject.toml           # project metadata + dependencies (managed by uv)
|-- uv.lock                  # locked dependency versions
|-- makefile                 # developer tasks (env, check, lint, pkg, docs, clean)
|-- docs/                    # Sphinx configuration and API reference
`-- pyDeepLearn/             # the importable package
    |-- LayerInterface.py          # abstract Layer base class
    |-- objectiveFuncInterface.py  # abstract objective function base class
    |-- InputLayer.py              # z-score normalization
    |-- FullyConnectedLayer.py     # dense layer + RecurrentFcLayer
    |-- LinearLayer.py             # identity / gradient pass-through
    |-- ReLuLayer.py
    |-- SigmoidLayer.py
    |-- TanhLayer.py
    |-- SoftmaxLayer.py
    |-- LeastSquares.py
    |-- LogLoss.py
    |-- CrossEntropy.py
    |-- RunUtilities.py            # training / evaluation helpers
    `-- tests/                     # unittest suite + test data
```

## Architecture

Every layer implements the `Layer` interface:

| Method | Purpose |
| --- | --- |
| `forward(dataIn)` | Apply the layer's transformation and cache its input/output. |
| `gradient()` | Return the layer's local Jacobian. |
| `backward(gradIn)` | Multiply the incoming gradient by the local gradient. |

Every objective function implements the `objectiveFuncInterface`:

| Method | Purpose |
| --- | --- |
| `eval(y, yhat)` | Score a batch of predictions. |
| `gradient(y, yhat)` | Return the derivative that seeds back propagation. |

A model is simply an ordered `list` of layers whose final entry is an objective
function:

```python
layers = [InputLayer(X_train), FullyConnectedLayer(4, 16), LinearLayer(),
          FullyConnectedLayer(16, 3), LinearLayer(), LeastSquares()]
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python
version, the virtual environment, and all dependencies. There is no
`requirements.txt`; dependencies are declared in `pyproject.toml` and pinned in
`uv.lock`.

1. Install uv:

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Create the environment and install everything (runtime plus the `dev` and
   `docs` dependency groups):

   ```bash
   uv sync --all-groups
   ```

   Or use the makefile shortcut:

   ```bash
   make env
   ```

`uv sync --all-groups` creates `.venv`, installs the package in editable mode,
and installs the `dev` and `docs` groups.

## Quick start

### Command line demonstrations

Run the two linear-regression demonstrations on CSV track data:

```bash
uv run python main.py -f1 ~/tracks/track1.csv -f2 ~/tracks/track2.csv -p
```

Train the recurrent model on a directory of tracks:

```bash
uv run python main.py -d ~/tracks
```

Reuse pre-trained weights instead of training:

```bash
uv run python main.py -d ~/tracks -w
```

`-p` displays the generated plots and `-v` enables verbose (debug) logging. Run
`uv run python main.py --help` for the full argument list.

### Library usage

```python
import numpy as np

from pyDeepLearn.InputLayer import InputLayer
from pyDeepLearn.FullyConnectedLayer import FullyConnectedLayer
from pyDeepLearn.LinearLayer import LinearLayer
from pyDeepLearn.LeastSquares import LeastSquares
from pyDeepLearn.RunUtilities import run_layers

X_train = np.random.random((100, 4))
Y_train = np.random.random((100, 2))

layers = [
    InputLayer(X_train),
    FullyConnectedLayer(X_train.shape[1], 16),
    LinearLayer(),
    FullyConnectedLayer(16, Y_train.shape[1]),
    LinearLayer(),
    LeastSquares(),
]
run_layers(layers, X_train, Y_train, max_epoch=100)
```

## Running the unit tests

```bash
uv run python -m unittest discover -s pyDeepLearn
```

Or with the makefile:

```bash
make check
```

## Linting

```bash
make lint          # equivalent to: uv run flake8 .
```

## Building the package

```bash
make pkg           # equivalent to: uv build
```

The wheel and source distribution are written to `dist/`. Install the built
wheel with:

```bash
uv pip install dist/pydeeplearn-1.0-py3-none-any.whl
```

## Generating the API documentation

The API reference is generated from the source docstrings with
[Sphinx](https://www.sphinx-doc.org/) using the `autodoc` and `napoleon`
extensions (NumPy-style docstrings):

```bash
make docs
```

The HTML output is written to `docs/_build/html/index.html`. The same build can
be run directly:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

## makefile targets

| Target | Description |
| --- | --- |
| `make env` | `uv sync --all-groups` -- create/refresh the environment. |
| `make check` | Run the unit test suite. |
| `make lint` | Run flake8. |
| `make pkg` | Build the wheel and sdist into `dist/`. |
| `make docs` | Build the Sphinx HTML API documentation. |
| `make clean` | Remove build artifacts, `.venv`, and caches. |

## Contributing

1. Create a branch for your change.
2. Install the environment with `uv sync --all-groups`.
3. Add or update tests and run `make check`.
4. Update the docstrings -- the Sphinx API documentation is generated from them
   with `make docs`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.

