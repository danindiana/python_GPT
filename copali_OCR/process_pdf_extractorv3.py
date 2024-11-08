import os
import io
import pytesseract
import fitz  # PyMuPDF for direct text extraction and as a secondary image extraction fallback
from PIL import Image, ImageOps
from pdf2image import convert_from_path  # New import for page-to-image conversion
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Set TESSDATA_PREFIX if needed
os.environ["TESSDATA_PREFIX"] = "/usr/local/share/"

def preprocess_image_for_ocr(image):
    """Preprocess the image for better OCR accuracy."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Increase contrast
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary threshold
    return image

def extract_text_without_ocr(pdf_path):
    """Attempt to extract embedded text directly from the PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text("text")  # Direct text extraction
    return text

def convert_pdf_to_images(pdf_path, dpi=300):
    """Convert each page of the PDF to a high-resolution image."""
    images = convert_from_path(pdf_path, dpi=dpi)
    return images

def extract_images_and_text_ocr(pdf_path, resize_factor=2):
    """
    Attempt to extract text from a PDF using direct extraction first.
    If direct extraction fails, fallback to OCR on each page image.
    """
    images = []  # Initialize an empty list for images
    pdf_text = extract_text_without_ocr(pdf_path)
    
    # If direct text extraction succeeded, return it
    if pdf_text.strip():
        return images, pdf_text, pdf_text  # `images` will be an empty list if no images were processed

    # Attempt to convert each PDF page into an image (fallback approach)
    print("Converting PDF pages to images for OCR...")
    images = convert_pdf_to_images(pdf_path)  # Convert all pages to images
    ocr_text = ""

    for page_number, pil_image in enumerate(images):
        # Resize image to save memory
        new_width = int(pil_image.width // resize_factor)
        new_height = int(pil_image.height // resize_factor)
        pil_image = pil_image.resize((new_width, new_height))
        
        # Preprocess image and run OCR
        processed_image = preprocess_image_for_ocr(pil_image)
        page_ocr_text = pytesseract.image_to_string(processed_image)
        ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text

    return images, "", ocr_text  # Return OCR text if direct extraction fails

# Set the primary GPU (use cuda:0 for consistency, but switch to cuda:1 if you prefer)
device = torch.device("cuda:0")

# Load ColQwen2 Model and Processor, placing the entire model on one GPU
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.float16,
).to(device).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Find PDF files in the current directory
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

if not pdf_files:
    print("No PDF files found in the current directory.")
    exit()

# Display PDF files with a numeric menu
print("Available PDF files:")
for i, pdf_file in enumerate(pdf_files):
    print(f"{i+1}. {pdf_file}")

# Get user input for PDF selection
selected_indices = input("Enter the number(s) of the PDF file(s) to process (comma-separated): ")
try:
    selected_indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
    selected_pdfs = [pdf_files[i] for i in selected_indices]
except (ValueError, IndexError):
    print("Invalid input. Please enter valid number(s).")
    exit()

# Process the selected PDF files with batching
batch_size = 2  # Adjust according to available memory
for pdf_path in selected_pdfs:
    images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)

    print(f"Processing images for {pdf_path}...")

    # Save OCR-like text to a file
    output_file = f"{pdf_path}_ocr_output.txt"
    with open(output_file, "w") as f:
        f.write("OCR-like extracted text:\n")
        f.write(ocr_text)

    print(f"\nOCR-like extracted text saved to {output_file}")

    # Process images in batches to prevent out-of-memory errors
    all_image_embeddings = []
    if images:
        for i in range(0, len(images), batch_size):
            image_batch = images[i:i + batch_size]
            batch_images = processor.process_images(image_batch).to(device)
            
            with torch.no_grad():
                image_embeddings = model(**batch_images)
                all_image_embeddings.append(image_embeddings)
            
            torch.cuda.empty_cache()

        if all_image_embeddings:
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        else:
            all_image_embeddings = None  # Explicitly set to None if no embeddings were created
            print("No image embeddings were created.")
    else:
        all_image_embeddings = None  # Set to None if no images were found
        print("No images found in the PDF for processing.")

    # Use OCR text if direct text extraction was empty
    if not pdf_text.strip() and ocr_text.strip():
        pdf_text = ocr_text  # Fallback to OCR text if direct extraction failed

    # Check if there is text content to process
    if pdf_text.strip():
        print("Processing text...")
        queries = [pdf_text]
        batch_queries = processor.process_queries(queries).to(device)
        
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
    else:
        query_embeddings = None  # Set to None if no text found
        print("No text found in the PDF for processing.")

    # Perform similarity calculation only if both image and text embeddings are available
    if all_image_embeddings is not None and query_embeddings is not None and len(all_image_embeddings) > 0:
        print("Calculating similarity scores...")
        scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
        
        print(f"Processed {pdf_path}:")
        for i, score in enumerate(scores[0]):
            print(f"  Image {i+1}: Similarity Score = {score:.4f}")
    else:
        print("Skipping similarity calculation due to missing image or text embeddings.")

# Final memory cleanup
torch.cuda.empty_cache()
