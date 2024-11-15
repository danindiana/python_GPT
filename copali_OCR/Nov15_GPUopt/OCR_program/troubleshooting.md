```markdown
# Copali PDF OCR Processor: Comprehensive Troubleshooting Guide

Welcome to the **Copali PDF OCR Processor** troubleshooting guide. This document consolidates all debugging steps and solutions for issues encountered during installation and runtime. Follow this guide to ensure a smooth setup and operation of the processor.

---

## Table of Contents

1. [Overview](#overview)
2. [Common Issues and Solutions](#common-issues-and-solutions)
   - [Virtual Environment Setup](#virtual-environment-setup)
   - [PyMuPDF (`fitz`) Module Conflict](#pymupdf-fitz-module-conflict)
   - [Tesseract OCR Configuration](#tesseract-ocr-configuration)
   - [GPU Selection and CUDA Memory Errors](#gpu-selection-and-cuda-memory-errors)
3. [Final Checklist](#final-checklist)
4. [FAQs and Support](#faqs-and-support)

---

## Overview

The **Copali PDF OCR Processor** is a tool designed to process PDFs by extracting text and images using advanced OCR and deep learning techniques. It leverages:
- **PyMuPDF** for PDF text/image extraction.
- **Tesseract OCR** for optical character recognition.
- **ColQwen2** for deep learning models.
- **CUDA** for GPU acceleration.

Misconfiguration or outdated dependencies can cause installation and runtime issues. This guide addresses these problems systematically.

---

## Common Issues and Solutions

### 1. Virtual Environment Setup

**Problem**: Dependencies conflict with globally installed packages or the environment is not activated.

**Solution**:
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the virtual environment is active before running the script:
   ```bash
   source venv/bin/activate
   ```

---

### 2. PyMuPDF (`fitz`) Module Conflict

**Problem**: Error `module 'fitz' has no attribute 'Document'` occurs due to a conflicting `fitz` package (`0.0.1.dev2`).

**Solution**:
1. Check for conflicting `fitz` packages:
   ```bash
   pip list | grep fitz
   ```
2. If `fitz 0.0.1.dev2` is installed, uninstall it:
   ```bash
   pip uninstall fitz
   ```
3. Reinstall PyMuPDF:
   ```bash
   pip install --force-reinstall pymupdf
   ```
4. Verify the `fitz` module:
   ```python
   import fitz
   doc = fitz.Document("/path/to/sample.pdf")
   print(doc[0].get_text())
   ```

---

### 3. Tesseract OCR Configuration

**Problem**: Error `Tesseract couldn't load any languages!` due to a missing or incorrectly configured `TESSDATA_PREFIX`.

**Solution**:
1. Locate the `tessdata` directory:
   ```bash
   find /usr/share -type d -name "tessdata"
   ```
   Example result: `/usr/share/tesseract-ocr/5/tessdata`.

2. Update `TESSDATA_PREFIX`:
   - Temporarily:
     ```bash
     export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"
     ```
   - Permanently (add to `~/.bashrc`):
     ```bash
     echo 'export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/"' >> ~/.bashrc
     source ~/.bashrc
     ```

3. Test Tesseract:
   ```bash
   tesseract /path/to/test_image.png output
   cat output.txt
   ```

4. Set `TESSDATA_PREFIX` in your script:
   ```python
   import os
   os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'
   ```

---

### 4. GPU Selection and CUDA Memory Errors

**Problem**: CUDA memory errors or improper GPU selection.

**Solution**:
1. Check available GPUs:
   ```bash
   nvidia-smi
   ```
2. Select the correct GPU in the script:
   ```python
   import torch
   torch.cuda.set_device(selected_gpu)  # Ensure selected_gpu is set correctly
   ```
3. Reduce batch size to prevent memory overflow:
   ```python
   batch_size = 1
   ```

---

## Final Checklist

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
   - Verify `eng.traineddata` exists:
     ```bash
     ls /usr/share/tesseract-ocr/5/tessdata/eng.traineddata
     ```
   - Test Tesseract CLI:
     ```bash
     tesseract /path/to/test_image.png output
     ```

4. **GPU Configuration**:
   - Check GPUs:
     ```bash
     nvidia-smi
     ```
   - Set GPU in the script:
     ```python
     torch.cuda.set_device(selected_gpu)
     ```
   - Reduce batch size if memory errors occur:
     ```python
     batch_size = 1
     ```

---

## FAQs and Support

**Q: Why is `fitz.Document()` not recognized?**  
A: A conflicting `fitz` package is installed. Follow the steps in [PyMuPDF (`fitz`) Module Conflict](#pymupdf-fitz-module-conflict).

**Q: Why does Tesseract fail to load `eng.traineddata`?**  
A: `TESSDATA_PREFIX` is misconfigured. Ensure it points to the correct `tessdata` directory as outlined in [Tesseract OCR Configuration](#tesseract-ocr-configuration).

**Q: How do I fix CUDA memory errors?**  
A: Reduce batch size and verify GPU configuration in [GPU Selection and CUDA Memory Errors](#gpu-selection-and-cuda-memory-errors).

---
