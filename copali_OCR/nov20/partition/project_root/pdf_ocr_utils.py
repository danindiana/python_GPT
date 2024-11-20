import pytesseract
from PIL import Image
from pypdfium2 import PdfDocument
import pymupdf_utils  # Import the pymupdf utility functions

def extract_images_and_text_ocr(pdf_path, resize_factor=2):
    """Extract images and text from PDF using OCR if necessary."""
    images = []
    pdf_text = pymupdf_utils.extract_text_without_ocr(pdf_path)  # Use the imported function

    if pdf_text.strip():
        return images, pdf_text, pdf_text  # `images` will be an empty list if no images were processed

    try:
        pdf = PdfDocument(pdf_path)
    except Exception as e:
        print(f"Failed to load PDF {pdf_path}: {e}")
        return [], "", ""

    ocr_text = ""

    for page_number, page in enumerate(pdf):
        width, height = page.get_size()
        bitmap = page.render()

        try:
            pil_image = bitmap.to_pil()
        except AttributeError:
            pixmap = bitmap.to_pixmap()
            pil_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

        new_width = int(width // resize_factor)
        new_height = int(height // resize_factor)
        pil_image = pil_image.resize((new_width, new_height))

        processed_image = preprocess_image_for_ocr(pil_image)
        try:
            page_ocr_text = pytesseract.image_to_string(processed_image)
        except pytesseract.TesseractError as e:
            print(f"Tesseract OCR failed for page {page_number} in {pdf_file}: {e}")
            page_ocr_text = ""

        ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text
        images.append(pil_image)

    return images, "", ocr_text
