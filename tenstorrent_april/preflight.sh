#!/bin/bash

# ============================================================
# PyBuda Installation Pre-flight Check Script
# ============================================================
# Checks system prerequisites and configuration before attempting
# the main PyBuda installation process.
# ============================================================

# --- Configuration Variables (Should match your intended install setup) ---
CHECK_PYTHON_VERSION="3.10"
CHECK_PYBUDA_SRC_DIR="$HOME/fresh-tt-buda" # Or wherever your tt-buda source is
CHECK_KMD_SRC_DIR="$HOME/tt-kmd" # Or wherever your tt-kmd source is
CHECK_SYSTEM_TOOLS_DIR="$HOME/tt-system-tools" # Or wherever tt-system-tools is
# Set this to the KMD version required by the PyBuda version you plan to install
CHECK_REQUIRED_KMD_VERSION="1.33" # <-- EXAMPLE! CHECK DOCS!

# --- Helper Function ---
check_command() {
  if command -v "$1" &> /dev/null; then
    echo "[PASS] Command '$1' found."
    if [ "$#" -gt 1 ]; then
      shift
      echo "       Version: $("$@")"
    fi
  else
    echo "[FAIL] Command '$1' not found in PATH."
    return 1
  fi
  return 0
}

check_package() {
  if dpkg -s "$1" &> /dev/null && dpkg -s "$1" | grep "Status: install ok installed" &> /dev/null; then
    echo "[PASS] Package '$1' is installed."
  else
    echo "[FAIL] Package '$1' is not installed or dpkg check failed."
    return 1
  fi
  return 0
}

check_dir() {
  if [ -d "$1" ]; then
    echo "[PASS] Directory '$1' exists."
  else
    echo "[FAIL] Directory '$1' not found."
    return 1
  fi
  return 0
}

# --- Start Checks ---
FAILED_CHECKS=0
echo "========================================"
echo "== Starting Pre-flight Checks..."
echo "========================================"

echo "--- OS Check ---"
OS_ID=$(lsb_release -is 2>/dev/null)
OS_REL=$(lsb_release -rs 2>/dev/null)
if [ "$OS_ID" == "Ubuntu" ] && [[ "$OS_REL" == "22.04"* ]]; then
  echo "[PASS] OS is Ubuntu 22.04 ($OS_REL)."
else
  echo "[FAIL] OS is not Ubuntu 22.04 (Found: ${OS_ID:-unknown} ${OS_REL:-unknown})."
  ((FAILED_CHECKS++))
fi

echo "--- Sudo Check ---"
if sudo -n true 2>/dev/null; then
  echo "[PASS] User has passwordless sudo access (or recently used sudo)."
else
  if sudo -v -A 2>/dev/null; then
     echo "[PASS] User can likely elevate using sudo (password may be prompted)."
  else
     echo "[WARN] Could not verify sudo access non-interactively. Install script needs sudo."
     # Not failing the check, as password prompt might work
  fi
fi


echo "--- Required Tools Check ---"
check_command "python${CHECK_PYTHON_VERSION}" "python${CHECK_PYTHON_VERSION}" --version || ((FAILED_CHECKS++))
check_command "git" git --version || ((FAILED_CHECKS++))
check_command "g++" g++ --version || ((FAILED_CHECKS++))
check_command "cmake" cmake --version || ((FAILED_CHECKS++))
check_command "dkms" dkms --version || ((FAILED_CHECKS++))
check_command "make" make --version || ((FAILED_CHECKS++))

echo "--- Required Packages Check ---"
check_package "build-essential" || ((FAILED_CHECKS++))
check_package "python${CHECK_PYTHON_VERSION}-venv" || ((FAILED_CHECKS++))
check_package "libgtest-dev" || ((FAILED_CHECKS++))

echo "--- Source Directory Checks ---"
check_dir "${CHECK_PYBUDA_SRC_DIR}" || ((FAILED_CHECKS++))
check_dir "${CHECK_KMD_SRC_DIR}" || ((FAILED_CHECKS++))
check_dir "${CHECK_SYSTEM_TOOLS_DIR}" || ((FAILED_CHECKS++))

echo "--- HugePages Check ---"
HP_TOTAL=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')
HP_SIZE_KB=$(grep Hugepagesize /proc/meminfo | awk '{print $2}')
if [ -z "$HP_TOTAL" ] || [ -z "$HP_SIZE_KB" ]; then
    echo "[FAIL] Could not read HugePages info from /proc/meminfo."
    ((FAILED_CHECKS++))
elif [ "$HP_TOTAL" -lt 100 ]; then # Example threshold: Check if less than ~100 pages allocated
    echo "[WARN] Low number of HugePages detected (Total: ${HP_TOTAL}). Tenstorrent usually requires more."
    echo "       Run ${CHECK_SYSTEM_TOOLS_DIR}/hugepages-setup.sh from Tenstorrent docs."
    # Might not be a hard fail, but good to warn
else
    TOTAL_MB=$(( HP_TOTAL * HP_SIZE_KB / 1024 ))
    echo "[PASS] HugePages detected (Total: ${HP_TOTAL}, Size: ${HP_SIZE_KB}kB, Approx Total: ${TOTAL_MB}MB)."
fi

echo "--- Current KMD Check ---"
CURRENT_KMD_VERSION=$(modinfo tenstorrent 2>/dev/null | grep '^version:' | awk '{print $2}')
if [ -z "$CURRENT_KMD_VERSION" ]; then
  echo "[INFO] No Tenstorrent KMD currently loaded."
  echo "       Install script will attempt to install v${CHECK_REQUIRED_KMD_VERSION}."
elif [ "$CURRENT_KMD_VERSION" == "$CHECK_REQUIRED_KMD_VERSION" ]; then
  echo "[PASS] Currently loaded KMD version (${CURRENT_KMD_VERSION}) matches required version."
else
  echo "[WARN] Currently loaded KMD version (${CURRENT_KMD_VERSION}) does NOT match required (${CHECK_REQUIRED_KMD_VERSION})."
  echo "       Install script will attempt to remove current and install required version."
fi
echo "       DKMS Status:"
dkms status | grep tenstorrent || echo "       No tenstorrent modules found in DKMS."


echo "========================================"
if [ "$FAILED_CHECKS" -eq 0 ]; then
  echo "== Pre-flight Check Summary: PASS"
  echo "== System appears to meet basic prerequisites."
  echo "== Remember to verify compatibility (PyBuda version, KMD, Firmware) from official docs!"
else
  echo "== Pre-flight Check Summary: FAIL"
  echo "== Found ${FAILED_CHECKS} potential issue(s) marked [FAIL] above."
  echo "== Please address these issues before running the installation script."
fi
echo "========================================"

exit $FAILED_CHECKS
