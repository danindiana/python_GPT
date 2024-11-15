```markdown
# Copali PDF OCR Processor: Troubleshooting and Debugging Guide

This guide outlines common issues and their solutions when installing, configuring, or running the **Copali PDF OCR Processor**. Follow the checklist to debug problems and get the system operational.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation Issues](#installation-issues)
    - Missing or Conflicting Dependencies
    - Virtual Environment Setup
3. [Runtime Issues](#runtime-issues)
    - PyMuPDF (`fitz`) Module Conflict
    - Tesseract OCR Configuration
    - GPU Selection and CUDA Memory Errors
4. [Final Checklist](#final-checklist)

---

## Overview

The Copali PDF OCR Processor uses several libraries, including:
- **PyMuPDF** (for PDF text/image extraction)
- **pytesseract** (for OCR)
- **ColQwen2** (for deep learning models)
- **NVIDIA CUDA** (for GPU acceleration)

Misconfiguration, outdated libraries, or conflicts between packages can lead to errors. This guide provides solutions for common problems encountered during installation and execution.

---

## Installation Issues

### 1. Missing or Conflicting Dependencies
- **Problem**: Required packages are not installed or conflict with other packages.
- **Solution**:
  - Create a virtual environment to isolate dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
  - Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Virtual Environment Setup
- **Problem**: The system uses global packages instead of the isolated environment.
- **Solution**:
  - Always activate the virtual environment before running the script:
    ```bash
    source venv/bin/activate
    ```
  - Check installed packages to ensure they are local:
    ```bash
    pip list
    ```

---

## Runtime Issues

### 1. PyMuPDF (`fitz`) Module Conflict
- **Problem**: Error `module 'fitz' has no attribute 'Document'` occurs.
- **Cause**: A conflicting `fitz` package (`0.0.1.dev2`) is installed instead of the `fitz` module from PyMuPDF.
- **Solution**:
  1. Check for conflicting packages:
     ```bash
     pip list | grep fitz
     ```
  2. Uninstall the incorrect `fitz` package:
     ```bash
     pip uninstall fitz
     ```
  3. Reinstall PyMuPDF:
     ```bash
     pip install --force-reinstall pymupdf
     ```
  4. Test `fitz` functionality:
     ```python
     import fitz
     doc = fitz.Document("/path/to/sample.pdf")
     print(doc[0].get_text())
     ```

---

### 2. Tesseract OCR Configuration
- **Problem**: Error `Tesseract couldn't load any languages! Could not initialize tesseract.` occurs.
- **Cause**: Tesseract is unable to find the `eng.traineddata` file or the `TESSDATA_PREFIX` environment variable is misconfigured.
- **Solution**:
  1. Verify `eng.traineddata` exists:
     ```bash
     ls /usr/share/tesseract-ocr/4.00/tessdata/
     ```
  2. Set `TESSDATA_PREFIX` in your shell and Python script:
     ```bash
     export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata/"
     ```
     In the script:
     ```python
     import os
     os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/'
     ```
  3. Test Tesseract independently:
     ```bash
     tesseract /path/to/test_image.png output
     cat output.txt
     ```
  4. Test Tesseract in Python:
     ```python
     from PIL import Image
     import pytesseract
     pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
     text = pytesseract.image_to_string(Image.open("/path/to/image.png"))
     print(text)
     ```

---

### 3. GPU Selection and CUDA Memory Errors
- **Problem**: CUDA memory errors or improper GPU selection.
- **Solution**:
  1. Check available GPUs using `nvidia-smi`:
     ```bash
     nvidia-smi
     ```
  2. In the script, set the correct GPU:
     ```python
     import torch
     torch.cuda.set_device(selected_gpu)  # Ensure selected_gpu is set correctly
     ```
  3. Reduce batch size to avoid memory overflow:
     ```python
     batch_size = 1  # Reduce processing size
     ```

---

## Final Checklist

### Before Running the Script:
1. **Environment**:
   - Activate the virtual environment:
     ```bash
     source venv/bin/activate
     ```
   - Verify installed packages:
     ```bash
     pip list
     ```

2. **PyMuPDF**:
   - Confirm no conflicting `fitz` packages are installed:
     ```bash
     pip list | grep fitz
     ```
   - Reinstall PyMuPDF if needed:
     ```bash
     pip install --force-reinstall pymupdf
     ```

3. **Tesseract**:
   - Ensure `TESSDATA_PREFIX` is correctly set:
     ```bash
     echo $TESSDATA_PREFIX
     ```
   - Test Tesseract independently:
     ```bash
     tesseract /path/to/test_image.png output
     ```

4. **GPU Configuration**:
   - Check GPU availability:
     ```bash
     nvidia-smi
     ```
   - Reduce batch size if memory errors occur:
     ```python
     batch_size = 1
     ```

### After Debugging:
- Run the script:
  ```bash
  python copali_pdf_ocrv5.py
  ```

---

### Need Help?
If errors persist, collect the following information:
1. Python version:
   ```bash
   python --version
   ```
2. Installed packages:
   ```bash
   pip list
   ```
3. Full error traceback from the script.

Submit this information for further assistance. Happy debugging!
```



Certainly! Here's a recapitulation of the issues we've encountered and the steps we've taken to debug the code, formatted as a README.md file. This can serve as an install/run error checklist and troubleshooting manual.

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

Certainly! Below is a comprehensive `README.md` file that summarizes the issues encountered and the steps taken to debug and resolve them. This can serve as a troubleshooting manual for future reference.

---

# Copali PDF OCR Project

## Overview
This project processes PDF files to extract text and images using OCR (Optical Character Recognition) and other machine learning models. The project uses Python 3.12.7 and various libraries including PyMuPDF, PyTorch, and Tesseract OCR.

## Installation and Setup

### Prerequisites
- Python 3.12.7
- Tesseract OCR
- PyMuPDF (fitz)
- PyTorch
- NVIDIA GPU with CUDA support

### Step-by-Step Installation

1. **Install Python 3.12.7**
   - Download and extract Python 3.12.7 source code:
     ```bash
     wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
     tar -xzf Python-3.12.7.tgz
     cd Python-3.12.7
     ```
   - Configure and build Python:
     ```bash
     ./configure --with-ensurepip=install --enable-shared --prefix=/usr/local LDFLAGS="-Wl,-rpath /usr/local/lib"
     make
     sudo make altinstall
     ```

2. **Install Tesseract OCR**
   - Install Tesseract OCR and the English language data:
     ```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr tesseract-ocr-eng
     ```

3. **Install PyMuPDF (fitz)**
   - Install PyMuPDF:
     ```bash
     pip install pymupdf
     ```

4. **Install PyTorch with CUDA Support**
   - Install PyTorch with CUDA support:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

5. **Set Environment Variables**
   - Set the `TESSDATA_PREFIX` environment variable:
     ```bash
     export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00
     ```
   - Add it to your shell profile (e.g., `.bashrc`, `.zshrc`):
     ```bash
     echo 'export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00' >> ~/.bashrc
     source ~/.bashrc
     ```

6. **Create and Activate Virtual Environment**
   - Create a virtual environment:
     ```bash
     python3.12 -m venv venv
     ```
   - Activate the virtual environment:
     ```bash
     source venv/bin/activate
     ```

7. **Install Project Dependencies**
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

## Troubleshooting Checklist

### 1. **ModuleNotFoundError: No module named '_ctypes'**
   - **Issue**: The `ctypes` module is missing.
   - **Solution**:
     1. Install `libffi-dev`:
        ```bash
        sudo apt-get install libffi-dev
        ```
     2. Rebuild Python 3.12.7:
        ```bash
        ./configure --with-ensurepip=install --enable-shared --prefix=/usr/local LDFLAGS="-Wl,-rpath /usr/local/lib"
        make
        sudo make altinstall
        ```
     3. Verify the `ctypes` module:
        ```bash
        /usr/local/bin/python3.12 -c "import ctypes; print(ctypes)"
        ```

### 2. **TesseractError: Error opening data file /usr/share/tesseract-ocr/4.00/eng.traineddata**
   - **Issue**: Tesseract OCR cannot find the `eng.traineddata` file.
   - **Solution**:
     1. Ensure `TESSDATA_PREFIX` is set correctly:
        ```bash
        export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00
        ```
     2. Verify the `eng.traineddata` file exists:
        ```bash
        ls $TESSDATA_PREFIX/tessdata/eng.traineddata
        ```
     3. Install the English language data if missing:
        ```bash
        sudo apt-get install tesseract-ocr-eng
        ```

### 3. **Failed to extract text from file: module 'fitz' has no attribute 'open'**
   - **Issue**: PyMuPDF (fitz) version incompatibility.
   - **Solution**:
     1. Upgrade PyMuPDF to the latest version:
        ```bash
        pip install --upgrade pymupdf
        ```
     2. Use `fitz.Document()` instead of `fitz.open()` in your script:
        ```python
        with fitz.Document(pdf_path) as doc:
            for page in doc:
                text += f"\n--- Page {page.number + 1} ---\n"
                text += page.get_text("text")
        ```

### 4. **CUDA Out-of-Memory Error**
   - **Issue**: The script runs out of GPU memory.
   - **Solution**:
     1. Reduce batch sizes in your script.
     2. Ensure proper memory cleanup:
        ```python
        torch.cuda.empty_cache()
        ```

### 5. **General Debugging Tips**
   - **Verify Python Installation**:
     ```bash
     python3.12 --version
     ```
   - **Verify Tesseract Installation**:
     ```bash
     tesseract --version
     ```
   - **Verify PyTorch Installation**:
     ```bash
     python -c "import torch; print(torch.__version__)"
     ```
   - **Verify GPU Availability**:
     ```bash
     nvidia-smi
     ```

## Running the Script

1. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
   ```

2. **Run the Script**
   ```bash
   python copali_ocr_gpu2.py
   ```

## Conclusion
This `README.md` provides a comprehensive guide to installing, setting up, and troubleshooting the Copali PDF OCR project. Follow the steps and use the troubleshooting checklist to resolve any issues that may arise during the installation and execution of the script.

---

Feel free to customize this `README.md` further to include any additional details specific to your project.
