This code is a comprehensive PDF processing tool that extracts text content from PDF files, with special attention to academic papers containing mathematical formulas and symbols. Here's a detailed explanation of how it works:

### 1. **Overview**
The script uses multiple OCR (Optical Character Recognition) models to extract text from PDFs, with fallback mechanisms for different content types:
- Native PDF text extraction (when the PDF contains selectable text)
- Specialized OCR models for academic papers (Nougat) and general text (TrOCR)
- Math OCR for equations and formulas (using LaTeX OCR)

### 2. **Key Components**

#### **OCR Models**
- **NougatOCR**: Specialized for academic papers (uses `facebook/nougat-base` model)
- **TrOCR**: General-purpose text recognition (uses `microsoft/trocr-large-printed`)
- **MathOCR**: For mathematical equations (uses `pix2tex` for LaTeX conversion)

#### **Supporting Functions**
- `clean_latex_text`: Converts LaTeX markup to readable plain text
- `clean_ocr_text`: Applies symbol corrections (e.g., fixing common OCR errors)
- `extract_page_content`: Combines multiple extraction methods for best results
- `process_pdf`: Processes an entire PDF file page by page

### 3. **Workflow**

#### **Initialization**
1. **Command-line arguments** are parsed to configure:
   - Input/output directories
   - OCR model selection (`nougat`, `trocr`, or `none`)
   - Math OCR enable/disable
   - GPU/CPU selection
   - LaTeX cleaning option

2. **OCR models are loaded** based on configuration:
   - Nougat or TrOCR model initialized with appropriate device (GPU/CPU)
   - Math OCR initialized if enabled

#### **PDF Processing**
For each PDF file:
1. The PDF is opened using `fitz` (PyMuPDF).
2. For each page:
   - **Native text extraction** is attempted first using `page.get_text()`.
   - If native text is insufficient (less than `MIN_TEXT_LENGTH`), **OCR is used**:
     - The page is rendered as an image at appropriate DPI (300 for Nougat, 200 for TrOCR)
     - The image is processed by the selected OCR model
   - If math-related keywords are detected, **Math OCR** is attempted
3. Results from all methods are combined and cleaned:
   - LaTeX markup is converted to plain text if requested
   - Common OCR errors are corrected using `SYMBOL_MAP`
4. Extracted content is saved to a text file in the output directory.

#### **Output**
- Each page's content is saved with clear section headers:
  - Native PDF text
  - OCR results (if different from native text)
  - Math LaTeX (if detected)
- Processing statistics are printed at completion.

### 4. **Special Features**

1. **Intelligent Fallback System**:
   - Uses native text when available
   - Falls back to OCR only when necessary
   - Special handling for math content

2. **Symbol Correction**:
   - Fixes common OCR errors (e.g., "D" → "Dθ", "dn(" → "dη(")
   - Converts LaTeX symbols to proper Unicode (e.g., `\alpha` → "α")

3. **GPU Optimization**:
   - Automatically detects and uses GPU if available
   - Allows manual GPU selection
   - Estimates VRAM usage

4. **Math Support**:
   - Detects math regions by analyzing image histograms
   - Uses specialized LaTeX OCR for equations
   - Can convert LaTeX to readable plain text

### 5. **Usage Example**
To process all PDFs in a directory using Nougat OCR with math support:
```bash
python script.py input_pdfs/ output_text/ --ocr-model nougat --device cuda
```

### 6. **Error Handling**
- Graceful degradation when models fail to load
- Error messages included in output when processing fails
- Clear warnings about missing dependencies

### 7. **Dependencies**
The code requires several specialized libraries:
- PyMuPDF (`fitz`) for PDF handling
- Transformers (Hugging Face) for Nougat and TrOCR
- PIL/Pillow for image handling
- pytesseract for fallback OCR
- pix2tex for math OCR
- PyTorch for model execution

This tool is particularly valuable for academic work where PDFs may contain:
- Complex mathematical notation
- Multi-format content (text, equations, diagrams)
- Poor quality scans or non-selectable text
