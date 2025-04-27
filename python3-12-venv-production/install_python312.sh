#!/bin/bash

# Python 3.12 Compilation and Optimization Script (Fixed)
# Run with: sudo bash install_python312.sh

set -e

PYTHON_VERSION="3.12.4"
INSTALL_DIR="/opt/python312"
NUM_CORES=$(nproc)

# Install dependencies
echo "[1] Installing dependencies..."
apt update
apt install -y \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev \
    liblzma-dev tk-dev wget

# Download and extract Python
echo "[2] Downloading Python $PYTHON_VERSION..."
cd /tmp
wget -q "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"
tar -xzf "Python-$PYTHON_VERSION.tgz"
cd "Python-$PYTHON_VERSION"

# Configure with optimizations
echo "[3] Configuring with optimizations..."
./configure \
    --prefix="$INSTALL_DIR" \
    --enable-optimizations \
    --with-lto \
    --enable-shared \
    --with-system-expat \
    --with-system-ffi \
    --with-ensurepip=install

# Compile and install
echo "[4] Compiling (using $NUM_CORES cores)..."
make -j "$NUM_CORES"
make altinstall

# Update shared library links
echo "[5] Updating library paths..."
echo "/opt/python312/lib" | sudo tee /etc/ld.so.conf.d/python312.conf
sudo ldconfig

# Add to PATH
echo "[6] Adding to PATH..."
echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" | sudo tee /etc/profile.d/python312.sh
chmod +x /etc/profile.d/python312.sh
source /etc/profile.d/python312.sh

# Verify
echo "[7] Verifying..."
python3.12 --version
pip3.12 --version

echo "✅ Python $PYTHON_VERSION installed at $INSTALL_DIR"
echo "Use 'python3.12' or 'pip3.12' directly."
