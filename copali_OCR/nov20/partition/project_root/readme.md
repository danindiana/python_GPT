### Explanation of the Program

This Python program processes PDF files by performing the following tasks:
1. **Extract Text and Images**:
   - Attempts to extract text directly from PDF files using `pymupdf`.
   - If direct extraction fails, it uses OCR (`Tesseract`) to extract text from rendered images of the PDF pages.
   
2. **Preprocess Images**:
   - Converts images to grayscale.
   - Enhances contrast and applies a binary threshold for better OCR accuracy.

3. **Process Images and Text Using a Model**:
   - Utilizes the `ColQwen2` model for embedding generation and similarity scoring between image and text data.

4. **Handle GPU Selection**:
   - Queries available GPUs and allows the user to select one for processing.

5. **Progress Tracking**:
   - Logs the files that have been processed to avoid reprocessing on subsequent runs.

6. **Error Handling**:
   - Logs errors encountered during processing to a file (`error.log`).

7. **Final Outputs**:
   - Extracted text is saved to a directory specified by the user.
   - Skipped files and any issues are reported to the user.

---

### How to Set Up the Program

#### 1. **Clone the Repository**
To get started, clone the repository from GitHub:
```bash
git clone https://github.com/danindiana/python_GPT.git
cd python_GPT/copali_OCR/nov20/partition/project_root/
```

#### 2. **Install Dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

#### 3. **Set Up Tesseract**
Ensure Tesseract OCR is installed on your system:
- **On Ubuntu**:
  ```bash
  sudo apt update
  sudo apt install tesseract-ocr
  sudo apt install libtesseract-dev
  ```
- **On macOS** (via Homebrew):
  ```bash
  brew install tesseract
  ```
- **On Windows**:
  - Download and install Tesseract from its [official site](https://github.com/tesseract-ocr/tesseract).

Update the `TESSDATA_PREFIX` environment variable if needed:
```bash
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/
```

#### 4. **Set Up CUDA for GPU Processing**
Ensure your system has a CUDA-compatible GPU with the required drivers and `torch` installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. **Set Up Directories**
Prepare the directories for input and output:
1. Place your PDF files in a target directory (e.g., `input_pdfs`).
2. Create an output directory for the processed text files (e.g., `output_texts`).

#### 6. **Run the Program**
Launch the program and follow the prompts:
```bash
python main_script.py
```
- **Input Directory**: Provide the path to the folder containing the PDF files.
- **Output Directory**: Specify the folder where extracted text should be saved.

---

### Folder Structure After Cloning
```plaintext
project_root/
├── main_script.py            # Main program
├── preprocessing.py          # Handles image and text preprocessing
├── pdf_ocr_utils.py          # OCR and PDF handling utilities
├── gpu_selection.py          # GPU selection utilities
├── model_utils.py            # Model loading and interaction
├── progress_tracker.py       # Tracks processed files
├── error_logger.py           # Logs errors to a file
├── utils/
│   ├── __init__.py           # Marks `utils` as a Python package
│   ├── pymupdf_utils.py      # PDF utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # Instructions for usage
```

---

### Troubleshooting
- **Missing Dependencies**: If a module like `torch` or `pytesseract` is missing, install it using `pip install <module_name>`.
- **File Permissions**: Ensure your script has permission to read/write to the input/output directories.
- **CUDA Issues**: Verify that your GPU drivers and CUDA toolkit are properly installed.

---

### Repository Setup on GitHub

If someone clones the repository:
1. They should navigate to the `project_root/` directory.
2. Install dependencies as described above.
3. Follow the usage instructions in the `README.md` file (add one to the repo if not already present). 

Let me know if you'd like help refining the repository further!
