import os
import pdfplumber
import pytesseract
import logging
from PIL import Image
import traceback

# Ensure pytesseract uses the correct installation path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def prompt_user_for_directories():
    """
    Prompt the user for source and target directory paths.
    """
    source_dir = input("Please enter the source directory path: ")
    target_dir = input("Please enter the target directory path: ")
    return source_dir, target_dir

def extract_text_from_page(page):
    """
    Extract text from a page.
    """
    text = page.extract_text()
    if text is None:
        text = ""
    return text

def extract_text_from_image(image):
    """
    Use OCR to extract text from a PIL image.
    """
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_path, target_dir):
    """
    Extract text from a PDF file. The content from all pages, including text and OCR results from images,
    is combined and written to a single text file in the target directory.
    """
    with pdfplumber.open(pdf_path) as pdf:
        # Get the basename for the output file
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        logging.info(f"Extracting text from PDF: {base_filename}.pdf")

        all_text_content = []  # List to store all text content
        
        for page_num, page in enumerate(pdf.pages):
            text_content = extract_text_from_page(page)
            text_content += " \n\n"  # Add space between page text and image text

            # Extract text from images on the page
            image = page.to_image().original.convert('RGB')
            ocr_text = extract_text_from_image(image)
            if ocr_text:  # Append only if OCR text is extracted
                text_content += "Image Text:\n" + ocr_text + "\n\n"

            all_text_content.append(text_content)

        # Combine all text into one string
        combined_text = "\n\n".join(all_text_content)

        # Write the combined text to a single file named after the PDF
        output_filename = os.path.join(target_dir, f"{base_filename}.txt")
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            output_file.write(combined_text)
            logging.info(f"Saved extracted text to {output_filename}")

def main():
    source_dir, target_dir = prompt_user_for_directories()

    # Validate source and target directories
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        logging.error(f"Source directory does not exist or is not a directory: {source_dir}")
        return

    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            logging.info(f"Created target directory: {target_dir}")
        except Exception as e:
            logging.error(f"Unable to create target directory: {e}")
            return

    # Attempt to read and list all files in the source directory for debugging
    try:
        file_list = os.listdir(source_dir)
        logging.info(f"All files found in source directory: {file_list}")
    except Exception as e:
        logging.error(f"Unable to read the source directory: {e}")
        return

    # Process PDF files in the source directory
    pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
    logging.info(f"Number of PDF files found: {len(pdf_files)}")
    if len(pdf_files) == 0:
        logging.info("No PDF files found in source directory.")
        return

    for filename in pdf_files:
        pdf_path = os.path.join(source_dir, filename)
        try:
            extract_text_from_pdf(pdf_path, target_dir)
            logging.info(f"Successfully extracted text from {filename}")
        except Exception as e:
            logging.error(f"Failed to extract text from {filename}: {e}")
            traceback.print_exc()  # Print the full traceback

    logging.info("Extraction completed. Check the target directory for the output files.")

if __name__ == "__main__":
    main()
