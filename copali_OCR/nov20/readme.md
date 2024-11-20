Summary
pymupdf_utils.py: Contains the extract_text_without_ocr function, which is responsible for extracting text directly from a PDF using pymupdf.

main_script.py: The main script that imports the extract_text_without_ocr function from pymupdf_utils.py and uses it to extract text from PDFs. The rest of the code remains largely unchanged, focusing on OCR, image processing, and text chunking.

This separation makes the code more modular and easier to maintain, especially if you need to update or extend the pymupdf-related functionality in the future.
