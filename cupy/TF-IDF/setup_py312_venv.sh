#!/bin/bash

# Default Python version and virtual environment directory
DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_VENV_DIR="venv"

# Parse command-line arguments for Python version and virtual environment directory
while getopts "p:d:" opt; do
  case $opt in
    p) PYTHON_VERSION="$OPTARG"
    ;;
    d) VENV_DIR="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1
    ;;
  esac
done

# Use default values if not provided
PYTHON_VERSION=${PYTHON_VERSION:-$DEFAULT_PYTHON_VERSION}
VENV_DIR=${VENV_DIR:-$DEFAULT_VENV_DIR}

# Check if the specified Python version is installed
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    echo "Python $PYTHON_VERSION is not installed. Attempting to install..."

    # Attempt to install Python (example for Ubuntu)
    sudo apt-get update
    sudo apt-get install -y python$PYTHON_VERSION

    # Check if installation was successful
    if ! command -v python$PYTHON_VERSION &> /dev/null; then
        echo "Failed to install Python $PYTHON_VERSION."
        exit 1
    fi
fi

# Create a virtual environment
python$PYTHON_VERSION -m venv $VENV_DIR
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Output confirmation
echo "Python $PYTHON_VERSION virtual environment set up and activated in directory '$VENV_DIR'."
