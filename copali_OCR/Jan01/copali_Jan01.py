import os
import pytesseract
import fitz  # PyMuPDF for direct text extraction
import torch
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
from colpali_engine.models import ColQwen2, ColQwen2Processor
import subprocess
import json
import gc
import signal
import time
import sys

# Set TESSDATA_PREFIX explicitly
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/"
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Verify TESSDATA_PREFIX and eng.traineddata file
def verify_tessdata():
    tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
    if not os.path.exists(tessdata_path):
        raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
    if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
        raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

verify_tessdata()

# Determine script's directory for consistent log file location
script_dir = os.path.dirname(os.path.abspath(__file__))
progress_file = os.path.join(script_dir, "processed_files.log")
error_log_file = os.path.join(script_dir, "error.log")

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Processing timed out")

# Preprocess image for better OCR
def preprocess_image_for_ocr(image):
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Increase contrast
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary threshold
    return image

# Retry mechanism for Tesseract OCR
def robust_ocr(image, retries=3):
    for attempt in range(retries):
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            print(f"OCR attempt {attempt + 1} failed: {e}")
            time.sleep(1)  # Wait before retrying
    raise RuntimeError("Tesseract OCR failed after multiple retries.")

# Extract text without OCR
def extract_text_without_ocr(pdf_path, timeout=30):
    text = ""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                print(f"Processing page {page_num + 1} of {pdf_path}")
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text("text")
    except TimeoutError:
        print(f"Processing timed out for file {pdf_path}. Skipping...")
        return ""
    except Exception as e:
        print(f"Failed to extract text from file {pdf_path}: {e}")
        return ""
    finally:
        signal.alarm(0)  # Cancel the alarm
    return text

# Extract images and text from PDF with OCR fallback
def extract_images_and_text_ocr(pdf_path, resize_factor=2):
    images = []
    pdf_text = extract_text_without_ocr(pdf_path)

    if pdf_text.strip():
        return images, pdf_text, pdf_text

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
            page_ocr_text = robust_ocr(processed_image)
        except RuntimeError as e:
            print(f"OCR failed for page {page_number} in {pdf_path}: {e}")
            page_ocr_text = ""

        ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text
        images.append(pil_image)

    return images, "", ocr_text

# Load and save progress
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()

def save_progress(processed_files):
    with open(progress_file, "w") as f:
        json.dump(list(processed_files), f)

# Fetch GPU information and select GPU
def get_gpu_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpus = []
    for line in output.strip().split('\n'):
        index, name, utilization = line.split(', ')
        gpus.append((int(index), name, int(utilization)))
    return gpus

def select_gpu(gpus):
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

# Split text into chunks
def split_text_into_chunks(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Main processing
def main():
    input_dir = input("Enter the path of the target directory containing PDF files: ")
    output_dir = input("Enter the path of the output directory for processed text files: ")

    if not os.path.isdir(input_dir):
        print("The target directory does not exist.")
        exit()
    if not os.path.isdir(output_dir):
        print("The output directory does not exist.")
        exit()

    processed_files = load_progress()
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf') and f not in processed_files]

    if not pdf_files:
        print("All PDF files in the directory have already been processed.")
        exit()

    resume_prompt = input("Do you want to resume from the last stop? (y/n): ").strip().lower()
    if resume_prompt != "y":
        print("Starting fresh. Clearing progress log...")
        processed_files = set()
        save_progress(processed_files)

    gpus = get_gpu_info()
    selected_gpu = select_gpu(gpus)
    torch.cuda.set_device(selected_gpu)

    device = torch.device(f"cuda:{selected_gpu}")
    model = ColQwen2.from_pretrained("vidore/colqwen2-v0.1", torch_dtype=torch.float16).to(device).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    max_chunk_size = 4300
    max_sequence_length = 32000
    skipped_files = []
    error_threshold = 5  # Maximum number of errors allowed before skipping a document

    try:
        for pdf_file in pdf_files:
            if len(pdf_file) > 200:
                print(f"Skipping file {pdf_file} due to file name length exceeding 200 characters.")
                skipped_files.append(pdf_file)
                continue

            pdf_path = os.path.join(input_dir, pdf_file)
            error_count = 0  # Initialize error counter for this document

            try:
                print(f"Processing file: {pdf_file}")
                images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)
                print(f"Extracted images and text for {pdf_file}")

                if not pdf_text.strip() and not ocr_text.strip():
                    print(f"Skipping file {pdf_file} due to OCR failure.")
                    skipped_files.append(pdf_file)
                    continue

                output_file = os.path.join(output_dir, f"{pdf_file}_ocr_output.txt")
                with open(output_file, "w") as f:
                    f.write("OCR-like extracted text:\n")
                    f.write(ocr_text)
                processed_files.add(pdf_file)
                save_progress(processed_files)
                print(f"Successfully processed: {pdf_file}")

                print(f"Processing images for {pdf_file}...")

                all_image_embeddings = []
                if images:
                    for i in range(0, len(images), 1):  # Batch size reduced to 1
                        image_batch = images[i:i + 1]
                        batch_images = processor.process_images(image_batch).to(device)

                        with torch.no_grad():
                            try:
                                print(f"Processing image batch {i} for {pdf_file}...")
                                image_embeddings = model(**batch_images)
                                all_image_embeddings.append(image_embeddings)
                            except Exception as e:
                                print(f"Error processing image batch {i} for {pdf_file}: {e}")
                                error_count += 1
                                if error_count > error_threshold:
                                    print(f"Skipping file {pdf_file} due to too many errors.")
                                    skipped_files.append(pdf_file)
                                    break
                                torch.cuda.empty_cache()

                        torch.cuda.empty_cache()

                    if all_image_embeddings:
                        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
                    else:
                        all_image_embeddings = None
                        print("No image embeddings were created.")
                else:
                    all_image_embeddings = None
                    print("No images found in the PDF for processing.")

                if not pdf_text.strip() and ocr_text.strip():
                    pdf_text = ocr_text

                if pdf_text.strip():
                    print("Processing text...")
                    text_chunks = split_text_into_chunks(pdf_text, max_chunk_size)
                    similarity_scores = []
                    skip_due_to_length = False

                    for chunk in text_chunks:
                        if len(chunk.split()) > max_sequence_length:
                            print(f"Skipping file {pdf_file} due to chunk length exceeding {max_sequence_length}")
                            skip_due_to_length = True
                            skipped_files.append(pdf_file)
                            break

                        try:
                            queries = [chunk]
                            batch_queries = processor.process_queries(queries).to(device)

                            with torch.no_grad():
                                try:
                                    print(f"Processing text chunk for {pdf_file}...")
                                    query_embeddings = model(**batch_queries)
                                    torch.cuda.empty_cache()

                                    if all_image_embeddings is not None:
                                        scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                                        similarity_scores.append(scores[0].mean().item())
                                except Exception as e:
                                    print(f"Error processing text chunk for {pdf_file}: {e}")
                                    error_count += 1
                                    if error_count > error_threshold:
                                        print(f"Skipping file {pdf_file} due to too many errors.")
                                        skipped_files.append(pdf_file)
                                        break
                                    torch.cuda.empty_cache()
                        except torch.cuda.OutOfMemoryError:
                            print("Skipping due to CUDA memory issue.")
                            torch.cuda.empty_cache()
                            skip_due_to_length = True
                            skipped_files.append(pdf_file)
                            break

                    if skip_due_to_length:
                        continue

                    if similarity_scores:
                        avg_score = sum(similarity_scores) / len(similarity_scores)
                        print(f"Average Similarity Score for {pdf_file}: {avg_score:.4f}")
                    else:
                        print("No similarity scores were calculated.")
                else:
                    print("No text found in the PDF for processing.")

                # Free memory after processing each file
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                error_count += 1
                if error_count > error_threshold:
                    print(f"Skipping file {pdf_file} due to too many errors.")
                    skipped_files.append(pdf_file)
                with open(error_log_file, "a") as error_log:
                    error_log.write(f"{pdf_file}: {e}\n")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        # Perform cleanup operations
        torch.cuda.empty_cache()
        gc.collect()
        sys.exit(0)
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # Display summary of processing
    print("\nProcessing completed!")
    if skipped_files:
        print("\nThe following files were skipped due to errors or constraints:")
        for skipped_file in skipped_files:
            print(f" - {skipped_file}")
        print("\nDetails of the errors can be found in the error log file located in:")
        print(f" - {error_log_file}")
    else:
        print("\nNo files were skipped. All PDF files have been successfully processed!")

    # Save the final progress
    save_progress(processed_files)

    print("\nSummary:")
    print(f" - Total files processed: {len(processed_files)}")
    print(f" - Total files skipped: {len(skipped_files)}")
    print(f" - Output files saved in: {output_dir}")

    print("\nThank you for using the PDF processing script. Goodbye!")

if __name__ == "__main__":
    main()
