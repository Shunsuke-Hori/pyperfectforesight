#!/bin/bash

# Setup script for DynareByPython conda environment
# This script creates a conda environment with all required dependencies

set -e  # Exit on error

# Configuration
ENV_NAME="${1:-pyperfectforesight}"
PYTHON_VERSION="3.11"

echo "========================================"
echo "DynareByPython Environment Setup"
echo "========================================"
echo ""
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Warning: Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Aborting setup"
        exit 0
    fi
fi

echo "Creating conda environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo ""
echo "Installing required packages..."

# Activate environment and install packages
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install core scientific computing packages
echo "  - Installing numpy..."
conda install -n "$ENV_NAME" numpy -y

echo "  - Installing scipy..."
conda install -n "$ENV_NAME" scipy -y

echo "  - Installing sympy..."
conda install -n "$ENV_NAME" sympy -y

echo "  - Installing matplotlib..."
conda install -n "$ENV_NAME" matplotlib -y

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "  python main.py"
echo ""
echo "To deactivate the environment later, run:"
echo "  conda deactivate"
echo ""
