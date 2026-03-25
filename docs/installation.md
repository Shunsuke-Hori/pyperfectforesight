# Installation

## Requirements

- Python >= 3.9
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- SymPy >= 1.9.0

## From source (development)

### 1. Clone the repository

```bash
git clone https://github.com/Shunsuke-Hori/pyperfectforesight.git
cd pyperfectforesight
```

### 2. Set up the conda environment

A convenience script is provided that creates a conda environment named `pyperfectforesight` with all required dependencies:

```bash
bash setup_conda_env.sh
conda activate pyperfectforesight
```

### 3. Install the package in development mode

```bash
pip install -e .
```

This installs `pyperfectforesight` as an editable package so that changes to the source are reflected immediately without reinstalling.

### 4. (Optional) Install dev dependencies

To run the test suite and examples you will also need Matplotlib and pytest:

```bash
pip install -e ".[dev]"
```

## With pip (when published)

Once the package is published to PyPI, you will be able to install it directly:

```bash
pip install pyperfectforesight
```

## Building the documentation

Install the documentation dependencies and run Sphinx:

```bash
pip install -e ".[docs]"
sphinx-build docs docs/_build/html -b html
```

Then open `docs/_build/html/index.html` in your browser.
