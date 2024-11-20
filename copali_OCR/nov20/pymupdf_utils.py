# pymupdf_utils.py

import pymupdf

def extract_text_without_ocr(pdf_path):
    """Attempt to extract embedded text directly from the PDF."""
    text = ""
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:  # Corrected to iterate over pages
            text += f"\n--- Page {page.number + 1} ---\n"
            text += page.get_text("text")  # Direct text extraction
    except pymupdf.FileDataError as e:
        print(f"MuPDF error with file {pdf_path}: {e}")
        return ""
    return text
