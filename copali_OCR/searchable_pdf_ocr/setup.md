### **What the Program Does**

This program processes a directory of PDF files to create **OCR-Enhanced Searchable PDFs**. The output is a new PDF for each input, where text is extracted from the original PDF (or images within it), recognized via Tesseract OCR, and embedded as a searchable text layer.

#### **Main Steps of the Program**:
1. **Input Validation**:
   - Ensures input and output directories exist.
   - Lists all PDF files in the input directory.

2. **GPU Setup**:
   - Detects available GPUs using `nvidia-smi`.
   - Prompts the user to select a GPU for processing (though OCR itself is CPU-bound in the script).

3. **PDF Processing**:
   - For each PDF:
     1. Extracts images from each page.
     2. Preprocesses the images (grayscale conversion, thresholding) to improve OCR accuracy.
     3. Uses Tesseract OCR to extract text and bounding box data from the images.
     4. Embeds the recognized text into the original PDF as a hidden, searchable layer.

4. **Timeout and Error Handling**:
   - Sets a timeout to skip files that take too long to process.
   - Logs skipped files with reasons (e.g., errors or timeouts).

5. **Output**:
   - Saves searchable PDFs in the output directory with `_searchable.pdf` appended to filenames.

---

### **How to Set Up the Program**

1. **Install Required Software**:
   - Install Python and essential tools:
     ```bash
     sudo apt update
     sudo apt install python3 python3-pip python3-venv tesseract-ocr
     ```
   - Install Tesseract's English language data:
     ```bash
     sudo apt install tesseract-ocr-eng
     ```

2. **Install Python Dependencies**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Install the required Python libraries:
     ```bash
     pip install pytesseract pymupdf pypdfium2 torch pillow
     ```

3. **Set Up Environment Variables**:
   - Configure `TESSDATA_PREFIX` to point to the Tesseract data directory:
     ```bash
     export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/
     ```
   - To make this permanent, add it to your shell configuration (`~/.bashrc` or `~/.zshrc`):
     ```bash
     echo 'export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/' >> ~/.bashrc
     source ~/.bashrc
     ```

4. **Prepare Input and Output Directories**:
   - Create a directory for your input PDFs:
     ```bash
     mkdir input_pdfs
     ```
   - Place your PDF files in this directory.
   - Create an output directory for the results:
     ```bash
     mkdir output_pdfs
     ```

5. **Run the Program**:
   - Execute the script:
     ```bash
     python your_script.py
     ```
   - Enter the paths to the input and output directories when prompted.

---

### **How to Debug the Program**

1. **Verify Tesseract Installation**:
   - Test Tesseract OCR on an image:
     ```bash
     tesseract sample_image.png output -l eng
     ```
   - If this fails, ensure `eng.traineddata` exists in `/usr/share/tesseract-ocr/5/tessdata`.

2. **Check Environment Variables**:
   - Confirm `TESSDATA_PREFIX` is set correctly:
     ```bash
     echo $TESSDATA_PREFIX
     ls $TESSDATA_PREFIX/tessdata
     ```

3. **Test Python Libraries**:
   - Run a small test script to verify dependencies:
     ```python
     import pytesseract, fitz, torch, pypdfium2, PIL
     print("Dependencies installed correctly.")
     ```
   - If any import fails, reinstall the missing library.

4. **Check Permissions**:
   - Ensure the script has read/write permissions for input/output directories and PDF files:
     ```bash
     chmod -R 755 input_pdfs output_pdfs
     ```

5. **Run in Debug Mode**:
   - Add print statements or use Python's `logging` module to trace execution, e.g.:
     ```python
     print(f"Processing {pdf_file}...")
     ```

6. **Common Error Fixes**:
   - **`Tesseract OCR error: Could not initialize tesseract`**:
     - Verify `TESSDATA_PREFIX`.
     - Reinstall Tesseract and its language data.
   - **`FileNotFoundError: eng.traineddata is missing`**:
     - Reinstall English language data:
       ```bash
       sudo apt install tesseract-ocr-eng
       ```
   - **Timeouts**:
     - Increase the timeout value in the script:
       ```python
       signal.alarm(600)  # 10 minutes
       ```
   - **Memory Issues**:
     - Reduce the image resolution before OCR:
       ```python
       resize_factor = 3  # Instead of 2
       ```

---

### **Testing the Program**
1. Place a sample PDF in the `input_pdfs` directory.
2. Run the script and inspect the output directory.
3. Open the generated searchable PDF in a viewer and test for text search and selection.

---
