#!/bin/bash

# Set the Python version you want to use
PYTHON_VERSION="3.12.7"

# Check if Python 3.12.7 is installed
if ! python$PYTHON_VERSION --version &> /dev/null
then
    echo "Python $PYTHON_VERSION is not installed."
    exit 1
fi

# Create a virtual environment in the current directory
python$PYTHON_VERSION -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install the specified packages
pip install \
  cupy-cuda12x==13.3.0 \
  fastrlock==0.8.2 \
  joblib==1.4.2 \
  numpy==2.1.2 \
  scikit-learn==1.5.2 \
  scipy==1.14.1 \
  threadpoolctl==3.5.0

# Output the installed packages
echo "Installed packages:"
pip freeze

# Output confirmation
echo "Python $PYTHON_VERSION virtual environment set up, activated, and packages installed."
