Issues we've encountered and the steps we've taken to debug the code, formatted as a README.md file. This can serve as an install/run error checklist and troubleshooting manual.

```markdown
# Copali PDF OCR Script Troubleshooting Manual

This document provides a checklist and troubleshooting steps for installing and running the Copali PDF OCR script.

## Prerequisites

1. **Python Installation:**
   - Ensure Python 3.12 is installed.
   - Install Python development headers and virtual environment support.

   ```bash
   sudo apt update
   sudo apt install python3.12 python3.12-dev python3.12-venv
   ```

2. **Virtual Environment:**
   - Create and activate a virtual environment.

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   - Install the required Python packages.

   ```bash
   pip install pytesseract fitz pypdfium2 colpali_engine torch
   ```

4. **Install Tesseract OCR:**
   - Install Tesseract OCR and language data files.

   ```bash
   sudo apt update
   sudo apt install tesseract-ocr
   ```

## Common Issues and Solutions

### 1. `ModuleNotFoundError: No module named '_ctypes'`

**Symptom:**
- The script fails to import the `torch` module due to a missing `_ctypes` module.

**Solution:**
- Reinstall Python to ensure all necessary modules are included.

  ```bash
  sudo apt update
  sudo apt install python3.12 python3.12-dev python3.12-venv
  ```

- Reinstall PyTorch.

  ```bash
  pip install --upgrade torch
  ```

- Verify the installation of the `_ctypes` module.

  Create a file named `test_ctypes.py`:

  ```python
  import ctypes
  print("ctypes module is available.")
  ```

  Run the script:

  ```bash
  python test_ctypes.py
  ```

### 2. `TesseractError: (1, 'Error opening data file /usr/share/tesseract-ocr/4.00/eng.traineddata')`

**Symptom:**
- Tesseract OCR fails to find the `eng.traineddata` file.

**Solution:**
- Verify the Tesseract data directory and the presence of the `eng.traineddata` file.

  ```bash
  ls /usr/share/tesseract-ocr/4.00/tessdata
  ```

- Set the `TESSDATA_PREFIX` environment variable correctly in the script.

  ```python
  import os
  os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00'
  ```

### 3. `module 'fitz' has no attribute 'open'`

**Symptom:**
- The script fails to open a PDF file using the `fitz` module.

**Solution:**
- Ensure that the `fitz` module (PyMuPDF) is correctly imported and used.

  ```python
  import fitz
  ```

- Use the correct method to open a PDF file.

  ```python
  with fitz.open(pdf_path) as doc:
      # Process the PDF document
  ```

### 4. `FileNotFoundError: The directory /usr/local/share/tessdata does not exist`

**Symptom:**
- The script fails to find the Tesseract data directory.

**Solution:**
- Verify the Tesseract data directory and set the `TESSDATA_PREFIX` environment variable correctly.

  ```bash
  ls /usr/share/tesseract-ocr/4.00/tessdata
  ```

  ```python
  os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00'
  ```

### 5. `RuntimeError: Directory 'static/' does not exist`

**Symptom:**
- The script fails due to a missing `static/` directory.

**Solution:**
- Create the `static/` directory if it does not exist.

  ```bash
  mkdir -p /home/smduck/programs/copali/static
  ```

### 6. `ModuleNotFoundError: No module named 'tools'`

**Symptom:**
- The script fails to import a module named `tools`.

**Solution:**
- Create a dummy `tools.py` file as a temporary workaround.

  ```bash
  touch /home/smduck/programs/copali/venv/lib/python3.12/site-packages/tools.py
  ```

## Running the Script

1. **Set Environment Variables:**
   - Ensure the `TESSDATA_PREFIX` environment variable is set correctly in the script.

   ```python
   import os
   os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00'
   ```

2. **Run the Script:**
   - Execute the script from the command line.

   ```bash
   python copali_ocr_gpu.py
   ```

## Additional Tips

- **Memory Management:**
  - Use `torch.cuda.empty_cache()` to free up GPU memory when necessary.

- **Error Handling:**
  - Add error handling to catch and log exceptions for easier debugging.

- **GPU Selection:**
  - Ensure the script prompts the user to select a GPU and sets the selected GPU for use with PyTorch.

By following these steps and troubleshooting tips, you should be able to resolve common issues and run the Copali PDF OCR script successfully.
```

This README.md file provides a comprehensive checklist and troubleshooting manual for installing and running the Copali PDF OCR script. It covers the prerequisites, common issues, and their solutions, as well as additional tips for running the script.
