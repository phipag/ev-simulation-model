# `ev-simulation-model`

This package implements a Gaussian Mixture Model to simulate charging session data of electric vehicles. In particular, the session duration [h] and the electricity demand [kWh] can be simulated.

Assume the following variables:
* $x_1$: Charging duration in hours
* <img src="https://render.githubusercontent.com/render/math?math=x_1%20%3A%20%5Ctext%7BCharging%20duration%20in%20hours%7D">
* <img src="https://render.githubusercontent.com/render/math?math=x_2%20%3A%20%5Ctext%7BElectricity%20demand%20in%20kWh%7D">
* <img src="https://render.githubusercontent.com/render/math?math=z%20%3A%20%5Ctext%7BPlugin%20hour%20of%20the%20day%7D">

The model can generate draws from the following distributions:
* The joint distribution: <img src="https://render.githubusercontent.com/render/math?math=p%28x_1%2Cx_2%29">
* The marginal distributions: <img src="https://render.githubusercontent.com/render/math?math=p%28x_1%29%2Cp%28x_2%29%2Cp%28z%29">
* The conditional distributiuons: <img src="https://render.githubusercontent.com/render/math?math=p%28x_1%7Cz%29%2Cp%28x_2%7Cz%29%2Cp%28x_1%2Cx_2%7Cz%29">

# Developer notes

This project uses the following tools to automate tedious tasks in the development process:

* `poetry`: Reproducible dependency management and packaging
* `tox`: Test and workflow automation against different Python environments
* `pytest`: Unittests
* `black`: Code formatting
* `isort`: Code imports formatting
* `flake8`: Code linting
* `mypy`: Static type checking

## Where is my `setup.py` / `setup.cfg`?

There is no `setup.py` / `setup.cfg` configuration file because this package is managed by Poetry's
build system. This follows [PEP 517](https://www.python.org/dev/peps/pep-0517/) where `setuptools` is no longer the
default build system for Python. Instead it is possible to configure build systems via `pyproject.toml`. Poetry provides
a consistent configuration solely based on the `pyproject.toml` file. Read more details
here: https://setuptools.pypa.io/en/latest/build_meta.html.

## Usage

### Installation

Install [`poetry`](https://python-poetry.org/) on your machine if not yet installed. Install the project dependencies
using:

```shell
poetry install
```

### Development process

You can invoke all workflows using the `tox` CLI:

```shell
tox -e format # Format code
tox -e lint # Lint code
tox -e format,lint # Format first and then lint
tox -e python3.9 # Run pytest tests for Python 3.9 environment

# Run all workflows in logical order:
# format -> lint -> pytest against all Python environments
tox
```

### Where to find the Python interpreter path?

It might be useful to find the Python interpreter path for integration with your IDE. Print the path using:

```shell
poetry env info --path
```

For usage from command-line directly run:

```shell
poetry run <command> # Will run any command within the Poetry Python virtual environment
poetry shell # Will start a new shell as child process (recommended)
source `poetry env info --path`/bin/activate # Activation in current shell
```
