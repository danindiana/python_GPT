The script `setup_py312_venv.sh` performs several steps to ensure that a Python 3.12.7 virtual environment is set up and activated. Here is a detailed breakdown:

1. **Default Values**:
   - Sets default values for the Python version (`3.12.7`) and the virtual environment directory (`venv`).

2. **Parse Command-Line Arguments**:
   - Uses `getopts` to parse optional command-line arguments for custom Python version (`-p`) and virtual environment directory (`-d`).

3. **Assign Default Values if Not Provided**:
   - If no custom values are provided via command-line arguments, it uses the default values.

4. **Function to Download and Build Python from Source**:
   - Defines a function `install_python_from_source` that:
     - Installs necessary dependencies for building Python from source.
     - Downloads the Python 3.12.7 source code from the official Python website.
     - Extracts the downloaded tarball.
     - Configures and builds Python from the source code.
     - Installs the new Python version using `make altinstall` to avoid conflicts with the system Python.

5. **Check if Python 3.12.7 is Installed**:
   - Checks if the specified Python version is installed using the `command -v` command.
   - If Python 3.12.7 is not installed, it calls the `install_python_from_source` function to install it.

6. **Create a Virtual Environment**:
   - Uses the installed Python 3.12.7 to create a virtual environment in the specified directory using the `venv` module.

7. **Activate the Virtual Environment**:
   - Activates the newly created virtual environment by sourcing the `activate` script in the virtual environment's `bin` directory.

8. **Output Confirmation**:
   - Prints a confirmation message indicating that the virtual environment has been successfully set up and activated.

This script ensures that Python 3.12.7 is available and used for creating a virtual environment, while avoiding changes to the system-wide Python installation.
