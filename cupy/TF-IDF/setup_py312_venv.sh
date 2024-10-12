#!/bin/bash

# Default Python version and virtual environment directory
DEFAULT_PYTHON_VERSION="3.12.7"
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

# Function to download and build Python from source
install_python_from_source() {
    PYTHON_VERSION=$1
    echo "Downloading and building Python $PYTHON_VERSION from source..."

    # Install dependencies for building Python
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

    # Download and extract Python source code
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
    tar -xvf Python-$PYTHON_VERSION.tgz

    # Build and install Python
    cd Python-$PYTHON_VERSION
    ./configure --enable-optimizations
    make -j 8
    sudo make altinstall

    # Clean up
    cd ..
    rm -rf Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz
}

# Check if the specified Python version is installed
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    echo "Python $PYTHON_VERSION is not installed. Attempting to install..."
    install_python_from_source $PYTHON_VERSION
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
