import os
import pytesseract
import fitz  # PyMuPDF for direct text extraction
import torch
from PIL import Image, ImageOps, ImageFilter
from pypdfium2 import PdfDocument
from colpali_engine.models import ColQwen2, ColQwen2Processor
import subprocess
import concurrent.futures
import logging
from tqdm import tqdm
import argparse
import time
import smtplib
from email.mime.text import MIMEText
from cryptography.fernet import Fernet
from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TESSDATA_PREFIX if needed
os.environ["TESSDATA_PREFIX"] = "/usr/local/share/"

# Verify TESSDATA_PREFIX and eng.traineddata file
tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Preprocess the image for better OCR accuracy."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Increase contrast
    image = image.filter(ImageFilter.MedianFilter)  # Apply noise reduction
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary threshold
    return image

def extract_text_without_ocr(pdf_path: str) -> str:
    """Attempt to extract embedded text directly from the PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:  # Corrected to iterate over pages
                text += f"\n--- Page {page.number + 1} ---\n"
                text += page.get_text("text")  # Direct text extraction
    except Exception as e:
        logger.error(f"Failed to extract text from file {pdf_path}: {e}")
    return text

def extract_images_and_text_ocr(pdf_path: str, resize_factor: int = 2) -> tuple:
    """Extract images and text from PDF using OCR if necessary."""
    images = []
    pdf_text = extract_text_without_ocr(pdf_path)

    if pdf_text.strip():
        return images, pdf_text, pdf_text  # `images` will be an empty list if no images were processed

    try:
        pdf = PdfDocument(pdf_path)
    except Exception as e:
        logger.error(f"Failed to load PDF {pdf_path}: {e}")
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
        page_ocr_text = pytesseract.image_to_string(processed_image)
        ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text
        images.append(pil_image)

    return images, "", ocr_text

def split_text_into_chunks(text: str, chunk_size: int) -> list:
    """Split text into chunks of the specified size."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_gpu_info() -> list:
    """Fetch GPU information using nvidia-smi."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpus = []
    for line in output.strip().split('\n'):
        index, name, utilization = line.split(', ')
        gpus.append((int(index), name, int(utilization)))
    return gpus

def select_gpu(gpus: list) -> int:
    """Prompt the user to select a GPU."""
    print("Available GPUs:")
    for i, (index, name, utilization) in enumerate(gpus):
        print(f"{i + 1}. GPU {index}: {name} (Utilization: {utilization}%)")

    while True:
        try:
            selection = int(input("Select the GPU you wish to use (enter the corresponding number): "))
            if 1 <= selection <= len(gpus):
                return gpus[selection - 1][0]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def process_pdf_file(pdf_file: str, input_dir: str, output_dir: str, device: torch.device, model: ColQwen2, processor: ColQwen2Processor, max_chunk_size: int, max_sequence_length: int) -> str:
    """Process a single PDF file."""
    pdf_path = os.path.join(input_dir, pdf_file)
    images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)

    logger.info(f"Processing images for {pdf_file}...")

    # Save OCR-like text to a file in the output directory
    output_file = os.path.join(output_dir, f"{pdf_file}_ocr_output.txt")
    with open(output_file, "w") as f:
        f.write("OCR-like extracted text:\n")
        f.write(ocr_text)

    logger.info(f"\nOCR-like extracted text saved to {output_file}")

    # Process images with a batch size of 1 to prevent out-of-memory errors
    all_image_embeddings = []
    if images:
        for i in range(0, len(images), 1):  # Batch size reduced to 1
            image_batch = images[i:i + 1]
            batch_images = processor.process_images(image_batch).to(device)

            with torch.no_grad():
                try:
                    logger.info(f"Processing image batch {i} for {pdf_file}...")
                    image_embeddings = model(**batch_images)
                    all_image_embeddings.append(image_embeddings)
                except Exception as e:
                    logger.error(f"Error processing image batch {i} for {pdf_file}: {e}")
                    torch.cuda.empty_cache()
                    break

            torch.cuda.empty_cache()

        if all_image_embeddings:
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        else:
            all_image_embeddings = None
            logger.info("No image embeddings were created.")
    else:
        all_image_embeddings = None
        logger.info("No images found in the PDF for processing.")

    # Use OCR text if direct text extraction was empty
    if not pdf_text.strip() and ocr_text.strip():
        pdf_text = ocr_text

    # Check if there is text content to process
    if pdf_text.strip():
        logger.info("Processing text...")
        # Dynamically split text into manageable chunks based on max_chunk_size
        text_chunks = split_text_into_chunks(pdf_text, max_chunk_size)
        similarity_scores = []
        skip_due_to_length = False

        for chunk in text_chunks:
            if len(chunk.split()) > max_sequence_length:
                logger.info(f"Skipping file {pdf_file} due to chunk length exceeding {max_sequence_length}")
                skip_due_to_length = True
                break

            try:
                # Proceed with model processing for valid chunks
                queries = [chunk]
                batch_queries = processor.process_queries(queries).to(device)

                with torch.no_grad():
                    try:
                        logger.info(f"Processing text chunk for {pdf_file}...")
                        query_embeddings = model(**batch_queries)
                        torch.cuda.empty_cache()

                        if all_image_embeddings is not None:
                            scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                            similarity_scores.append(scores[0].mean().item())
                    except Exception as e:
                        logger.error(f"Error processing text chunk for {pdf_file}: {e}")
                        torch.cuda.empty_cache()
                        break
            except torch.cuda.OutOfMemoryError:
                logger.error("Skipping due to CUDA memory issue.")
                torch.cuda.empty_cache()
                skip_due_to_length = True
                break

        if skip_due_to_length:
            return pdf_file

        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            logger.info(f"Average Similarity Score for {pdf_file}: {avg_score:.4f}")
        else:
            logger.info("No similarity scores were calculated.")
    else:
        logger.info("No text found in the PDF for processing.")

    return None

def retry(func, retries=3, delay=2):
    """Retry mechanism for transient errors."""
    for _ in range(retries):
        try:
            return func()
        except Exception as e:
            logger.warning(f"Transient error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("Max retries exceeded")

def send_email(subject, body, to_email):
    """Send email notifications for critical errors or completion of long-running tasks."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'your_email@example.com'
    msg['To'] = to_email
    with smtplib.SMTP('smtp.example.com') as server:
        server.sendmail('your_email@example.com', to_email, msg.as_string())

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Colpali OCR Program')
    parser.add_argument('--input', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', required=True, help='Output directory for processed text files')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # Verify the directories exist
    if not os.path.isdir(input_dir):
        logger.error("The target directory does not exist.")
        exit()
    if not os.path.isdir(output_dir):
        logger.error("The output directory does not exist.")
        exit()

    # Fetch GPU information and prompt user to select a GPU
    gpus = get_gpu_info()
    selected_gpu = select_gpu(gpus)
    torch.cuda.set_device(selected_gpu)

    # Load model and processor only after directory confirmation to delay GPU allocation
    device = torch.device(f"cuda:{selected_gpu}")
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.float16  # Ensure half-precision to save memory
    ).to(device).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    # Set a lower maximum chunk size for memory efficiency
    max_chunk_size = 5000  # Reduced to 5000 to avoid high memory usage
    max_sequence_length = 32768  # Define the max sequence length

    # Process all PDF files in the target directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    if not pdf_files:
        logger.error("No PDF files found in the specified directory.")
        exit()

    # Initialize a list to store skipped files
    skipped_files = []

    # Process each PDF file in the input directory using a thread pool
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_pdf_file, pdf_file, input_dir, output_dir, device, model, processor, max_chunk_size, max_sequence_length) for pdf_file in tqdm(pdf_files, desc="Processing PDFs")]
        for future in concurrent.futures.as_completed(futures):
            skipped_file = future.result()
            if skipped_file:
                skipped_files.append(skipped_file)

    # Final memory cleanup
    torch.cuda.empty_cache()

    # Display the list of skipped files
    if skipped_files:
        logger.info("\nThe following files were skipped:")
        for skipped_file in skipped_files:
            logger.info(skipped_file)
    else:
        logger.info("\nNo files were skipped.")

if __name__ == "__main__":
    main()
