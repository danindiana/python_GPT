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

# Output confirmation
echo "Python $PYTHON_VERSION virtual environment set up and activated."
