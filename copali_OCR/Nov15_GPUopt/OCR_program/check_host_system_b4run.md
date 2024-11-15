Here’s a Python script that collects and displays relevant system information to help users debug issues when running your program for the first time. The script covers Python version, operating system, installed packages, GPU configuration, Tesseract setup, and other critical dependencies.

```python
import os
import platform
import subprocess
import sys

def check_python_version():
    print(f"Python Version: {platform.python_version()}")
    if sys.version_info < (3, 10):
        print("[Warning] Python version is below 3.10. Consider upgrading to Python 3.10 or higher.")

def check_operating_system():
    os_name = platform.system()
    os_version = platform.version()
    distro = "N/A"
    try:
        import distro as distro_lib
        distro = distro_lib.linux_distribution(full_distribution_name=True)
    except ImportError:
        if os_name == "Linux":
            distro = "Install `distro` Python package for detailed OS info."

    print(f"Operating System: {os_name} {os_version}")
    if os_name == "Linux":
        print(f"Linux Distribution: {distro}")

def check_installed_packages():
    try:
        print("Installed Python Packages:")
        subprocess.run(["pip", "list"], check=True)
    except Exception as e:
        print(f"[Error] Could not list installed packages: {e}")

def check_gpu_setup():
    try:
        gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print("GPU Information:")
        print(gpu_info.stdout)
    except FileNotFoundError:
        print("[Warning] NVIDIA `nvidia-smi` not found. Ensure NVIDIA drivers are installed.")
    except Exception as e:
        print(f"[Error] Failed to retrieve GPU information: {e}")

def check_tesseract_setup():
    try:
        tesseract_version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        print("Tesseract OCR Version:")
        print(tesseract_version.stdout)
    except FileNotFoundError:
        print("[Warning] Tesseract is not installed. Install it to enable OCR functionality.")
    except Exception as e:
        print(f"[Error] Failed to retrieve Tesseract version: {e}")

    tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "Not Set")
    print(f"TESSDATA_PREFIX: {tessdata_prefix}")
    if tessdata_prefix == "Not Set":
        print("[Warning] TESSDATA_PREFIX environment variable is not set.")

def check_cuda_version():
    try:
        cuda_version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print("CUDA Version:")
        print(cuda_version.stdout)
    except FileNotFoundError:
        print("[Warning] CUDA Toolkit is not installed. Required for GPU acceleration.")
    except Exception as e:
        print(f"[Error] Failed to retrieve CUDA version: {e}")

def check_virtual_environment():
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        print("Virtual Environment: Active")
    else:
        print("[Warning] Virtual Environment: Not Active. Consider using a virtual environment.")

def main():
    print("=== System Information and Requirements Check ===")
    check_python_version()
    check_operating_system()
    check_virtual_environment()
    print("\n--- Dependency Checks ---")
    check_installed_packages()
    check_gpu_setup()
    check_cuda_version()
    print("\n--- Tesseract OCR Setup ---")
    check_tesseract_setup()
    print("\n[Info] Use this information to debug and configure your system properly.")

if __name__ == "__main__":
    main()
```

### **Usage**
1. Save the script to a file, e.g., `system_check.py`.
2. Run the script from the command line:
   ```bash
   python system_check.py
   ```

### **Features**
- Checks Python version and warns if it’s below 3.10.
- Displays OS and distribution details.
- Lists all installed Python packages.
- Retrieves GPU information via `nvidia-smi`.
- Verifies if Tesseract OCR is installed and checks the `TESSDATA_PREFIX` variable.
- Checks CUDA Toolkit version if available.
- Warns if a virtual environment is not active.

This script ensures the user has all the relevant system details to debug errors and make necessary adjustments for a smooth program setup.
