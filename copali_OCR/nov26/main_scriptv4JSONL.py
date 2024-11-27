import os
import torch
import gc
import json
from preprocessing import preprocess_image_for_ocr, split_text_into_chunks
from pdf_ocr_utils import extract_images_and_text_ocr
from gpu_selection import get_gpu_info, select_gpu
from model_utils import load_model_and_processor
from progress_tracker import load_progress, save_progress
from error_logger import log_error


# Set TESSDATA_PREFIX if needed
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

# Verify TESSDATA_PREFIX and eng.traineddata file
tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

# Ask the user for input and output directories
input_dir = input("Enter the path of the target directory containing PDF files: ")
output_dir = input("Enter the path of the output directory for processed JSONL files: ")

# Verify the directories exist
if not os.path.isdir(input_dir):
    print("The target directory does not exist.")
    exit()
if not os.path.isdir(output_dir):
    print("The output directory does not exist.")
    exit()

# Load progress at the beginning
processed_files = load_progress()

# Filter files to process
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf') and f not in processed_files]

if not pdf_files:
    print("All PDF files in the directory have already been processed.")
    exit()

print(f"Found {len(pdf_files)} files to process.")

# Prompt user to resume or start fresh
resume_prompt = input("Do you want to resume from the last stop? (y/n): ").strip().lower()
if resume_prompt != "y":
    print("Starting fresh. Clearing progress log...")
    processed_files = set()
    save_progress(processed_files)

# Fetch GPU information and prompt user to select a GPU
gpus = get_gpu_info()
selected_gpu = select_gpu(gpus)
torch.cuda.set_device(selected_gpu)

# Load model and processor only after directory confirmation to delay GPU allocation
device = torch.device(f"cuda:{selected_gpu}")
model, processor = load_model_and_processor(device)

# Set a lower maximum chunk size for memory efficiency
max_chunk_size = 5000
max_sequence_length = 32768
max_jsonl_size = int(0.89 * (1024**3))  # 89% of 1 GB

# Initialize variables for JSONL file management
current_jsonl_file_index = 0
current_jsonl_size = 0
current_jsonl_data = []

# Helper function to save JSONL file
def save_jsonl_file(jsonl_data, output_dir, file_index):
    output_file = os.path.join(output_dir, f"output_{file_index:03d}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved JSONL file: {output_file}")

# Process each PDF file in the input directory
skipped_files = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)

    try:
        images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)

        print(f"Processing file {pdf_file}...")

        # Combine extracted text for output
        combined_text = ocr_text if not pdf_text.strip() else pdf_text

        if not combined_text.strip():
            print(f"No text found in {pdf_file}. Skipping...")
            skipped_files.append(pdf_file)
            continue

        # Split combined text into smaller chunks
        text_chunks = split_text_into_chunks(combined_text, max_chunk_size)

        # Add each chunk to the current JSONL data
        for i, chunk in enumerate(text_chunks):
            chunk_metadata = {
                "file_name": pdf_file,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "content": chunk
            }
            chunk_size = len(json.dumps(chunk_metadata).encode("utf-8"))
            if current_jsonl_size + chunk_size > max_jsonl_size:
                # Save current JSONL file and reset
                save_jsonl_file(current_jsonl_data, output_dir, current_jsonl_file_index)
                current_jsonl_file_index += 1
                current_jsonl_size = 0
                current_jsonl_data = []

            current_jsonl_data.append(chunk_metadata)
            current_jsonl_size += chunk_size

        processed_files.add(pdf_file)
        save_progress(processed_files)
        print(f"Successfully processed: {pdf_file}")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        log_error(pdf_file, str(e))
        continue

# Save any remaining JSONL data
if current_jsonl_data:
    save_jsonl_file(current_jsonl_data, output_dir, current_jsonl_file_index)

# Final memory cleanup
torch.cuda.empty_cache()
gc.collect()

# Display the list of skipped files
if skipped_files:
    print("\nThe following files were skipped:")
    for skipped_file in skipped_files:
        print(skipped_file)
else:
    print("\nNo files were skipped.")
