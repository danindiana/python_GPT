### **What This Program Does**

The program processes PDF files to create OCR-enhanced, searchable PDFs. Here's an overview of its functionality:

1. **Preprocessing Images for OCR**:
   - Extracts images from each page of the PDF.
   - Converts images to grayscale and applies thresholding for better OCR accuracy.

2. **OCR with Bounding Box Extraction**:
   - Uses Tesseract to recognize text and capture bounding box coordinates (`left`, `top`, `width`, `height`).
   - The output is a dictionary containing the recognized text and their positions.

3. **Creating Searchable PDFs**:
   - Embeds the OCR text into the original PDF using PyMuPDF (`fitz`) by overlaying invisible text boxes at the positions indicated by the bounding boxes.

4. **GPU Utilization**:
   - Detects available GPUs and allows the user to select one for processing (though GPU usage is not applied in this particular OCR operation).

5. **Timeout and Error Handling**:
   - Implements a timeout mechanism to skip files that take too long to process.
   - Skips files with excessively long names or unhandled errors.

6. **Output**:
   - For each processed PDF, an OCR-enhanced version is saved in the output directory with `_searchable.pdf` appended to the original filename.
   - Text within the searchable PDF can be selected, copied, and searched using PDF viewers like Adobe Acrobat or browsers.

### **How It Works**

1. **Input and Output Directories**:
   - The user is prompted to enter the path of the target directory containing PDF files and the path of the output directory for processed text files.

2. **GPU Selection**:
   - The program fetches GPU information using `nvidia-smi` and prompts the user to select a GPU for processing.

3. **PDF Processing**:
   - For each PDF file in the input directory:
     - Extracts images from each page.
     - Preprocesses the images for OCR.
     - Performs OCR to extract text and bounding boxes.
     - Adjusts bounding box coordinates to match the original PDF dimensions.
     - Embeds the OCR text into the original PDF as a hidden, searchable layer.
     - Saves the resulting searchable PDF to the output directory.

4. **Error Handling and Logging**:
   - Skips files with excessively long names or processing errors.
   - Logs skipped files and reasons for skipping.

### **Setup Using Python Virtual Environment (venv) on Ubuntu 22.04**

#### **Step-by-Step Setup**

1. **Install Required System Packages**:
   - Update the package list and install necessary dependencies.

   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv tesseract-ocr libtesseract-dev libleptonica-dev libfreetype6-dev libpng-dev libjpeg-dev libopenjp2-7-dev zlib1g-dev
   ```

2. **Create a Python Virtual Environment**:
   - Create a directory for your project and navigate into it.

   ```bash
   mkdir ocr_project
   cd ocr_project
   ```

   - Create a virtual environment.

   ```bash
   python3 -m venv venv
   ```

   - Activate the virtual environment.

   ```bash
   source venv/bin/activate
   ```

3. **Install Required Python Packages**:
   - Install the necessary Python packages using `pip`.

   ```bash
   pip install pytesseract fitz pypdfium2 torch pillow
   ```

4. **Download and Install Tesseract Language Data**:
   - Ensure the Tesseract language data is installed.

   ```bash
   sudo apt install tesseract-ocr-eng
   ```

5. **Run the Script**:
   - Save the provided script as `ocr_enhanced_searchable_pdf_generator_v3.py`.
   - Run the script.

   ```bash
   python ocr_enhanced_searchable_pdf_generator_v3.py
   ```

### **Detailed Explanation of the Script**

1. **Environment Setup**:
   - Sets the `TESSDATA_PREFIX` environment variable to the Tesseract data directory.
   - Verifies the existence of the Tesseract data directory and the `eng.traineddata` file.

2. **Image Preprocessing**:
   - Converts images to grayscale, increases contrast, and applies binary thresholding.

3. **OCR with Bounding Box Extraction**:
   - Uses Tesseract to extract text and bounding box coordinates.

4. **Creating Searchable PDFs**:
   - Embeds the OCR text into the original PDF as a hidden, searchable layer.

5. **GPU Utilization**:
   - Fetches GPU information and allows the user to select a GPU.

6. **Timeout and Error Handling**:
   - Implements a timeout mechanism to skip files that take too long to process.
   - Skips files with excessively long names or unhandled errors.

7. **Output**:
   - Saves the resulting searchable PDF to the output directory.

### **Conclusion**

This script is designed to enhance PDFs by making them searchable through OCR. It processes each page of a PDF, extracts text using Tesseract, and embeds the text as a hidden layer in the original PDF. The setup involves creating a Python virtual environment, installing necessary dependencies, and running the script.


To explicitly set the `--tessdata-dir` flag, you need to modify the sections of your script where `pytesseract.image_to_data` or other OCR-related functions are called. Here's how and where to apply this modification:

### **Update the `extract_text_with_boxes` Function**

Locate this function in your script:

```python
def extract_text_with_boxes(image):
    """Extract text with bounding boxes."""
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        return data  # Dictionary containing text and bounding box information
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract OCR error: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}
```

Modify it to include the `--tessdata-dir` configuration:

```python
def extract_text_with_boxes(image):
    """Extract text with bounding boxes."""
    try:
        # Explicitly pass the tessdata directory
        custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata"'
        data = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
        return data  # Dictionary containing text and bounding box information
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract OCR error: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}
```

---

### **Update Other Functions Using Tesseract**

If you have any other calls to Tesseract in your script (e.g., `pytesseract.image_to_string`), apply the same change. For example:

```python
page_ocr_text = pytesseract.image_to_string(processed_image)
```

Change to:

```python
custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata"'
page_ocr_text = pytesseract.image_to_string(processed_image, config=custom_config)
```

---

### **Why This Works**

- The `--tessdata-dir` flag explicitly tells Tesseract where to look for the language files.
- This bypasses potential issues with the `TESSDATA_PREFIX` environment variable.

---

### **Summary of Changes**

1. Locate all Tesseract function calls in your script (`pytesseract.image_to_data`, `pytesseract.image_to_string`, etc.).
2. Add the `--tessdata-dir` configuration using the `config` parameter.
3. Ensure the path matches your trained data location (`/usr/share/tesseract-ocr/5/tessdata`).

After making these updates, your script should correctly find and use the `eng.traineddata` file. Let me know if you encounter further issues!
