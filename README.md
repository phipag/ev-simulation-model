# `ev-simulation-model`

This package implements a Gaussian Mixture Model to simulate charging session data of electric vehicles. In particular, the session duration [h] and the electricity demand [kWh] can be simulated.

Assume the following variables:

$d$: Charging duration in hours  
$e$: Electricity demand in kWh  
$z$: Plug-in hour of the day

The model can generate draws from the following distributions:

The joint distribution: $p(d,e)$  
The marginal distributions: $p(d)$, $p(e)$, $p(z)$  
The conditional distributions: $p(d|z)$, $p(e|z)$, $p(d,e|z)$

The distribution $p(d,e|z)$ is of particular interest because it allows to simulate tuples of charging duration and electricity demand given a certain plug-in hour of the day. The marginal distribution $p(z)$ can be leveraged in a population model where a fixed number of chargers is assumed to determin how many new vehicles plug-in in a certain time interval. For example, to calculate the plug-in probability between 16 and 17 o'clock you can calculate $\int_{16}^{17} p(z) dz$. Multiplying the resulting probability with the number of chargers in the populations then yields the absolute number of new plug-ins within this hour.

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
