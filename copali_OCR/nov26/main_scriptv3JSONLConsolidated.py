import os
import torch
import gc
import json
import csv
import gzip
import bz2
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
output_dir = input("Enter the path of the output directory for processed text files: ")

# Prompt the user for desired output format (txt, csv, json, jsonl)
output_format = input("Enter the desired output format (txt, csv, json, jsonl): ").strip().lower()
if output_format not in ['txt', 'csv', 'json', 'jsonl']:
    print(f"Unsupported format: {output_format}. Please use 'txt', 'csv', 'json', or 'jsonl'.")
    exit()

# Prompt for compression (gzip, bzip2, or none)
compression_type = input("Enter the compression type (none, gzip, bzip2): ").strip().lower()
if compression_type not in ['none', 'gzip', 'bzip2']:
    print(f"Unsupported compression type: {compression_type}. Please use 'none', 'gzip', or 'bzip2'.")
    exit()

# Ask user for chunk size
chunk_size_lines = int(input("Enter the desired chunk size in lines (e.g., 1000): ").strip())

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
max_chunk_size = 5000  # Reduced to 5000 to avoid high memory usage
max_sequence_length = 32768  # Define the max sequence length

# Initialize a list to store skipped files
skipped_files = []

# Function to compress a file after it's written
def compress_file(output_file, compression_type):
    if compression_type == 'gzip':
        with open(output_file, 'rb') as f_in:
            with gzip.open(output_file + '.gz', 'wb') as f_out:
                f_out.writelines(f_in)
        os.remove(output_file)
        print(f"Compressed and saved: {output_file + '.gz'}")
    elif compression_type == 'bzip2':
        with open(output_file, 'rb') as f_in:
            with bz2.open(output_file + '.bz2', 'wb') as f_out:
                f_out.writelines(f_in)
        os.remove(output_file)
        print(f"Compressed and saved: {output_file + '.bz2'}")
    else:
        print(f"Saved uncompressed: {output_file}")

# Process each PDF file in the input directory
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

        # Chunk the output
        output_chunk = []
        chunk_counter = 1

        # Split the text into chunks based on line count
        lines = combined_text.splitlines()
        for i in range(0, len(lines), chunk_size_lines):
            chunk = lines[i:i + chunk_size_lines]
            output_chunk.append("\n".join(chunk))

            # Write this chunk to the output file
            output_file_base = os.path.join(output_dir, f"ocr_output_{chunk_counter}")
            output_file = output_file_base + ('.jsonl' if output_format == 'jsonl' else '.txt')

            with open(output_file, 'w', encoding='utf-8') as output_file_obj:
                # Write chunk data in the selected format
                if output_format == 'txt':
                    output_file_obj.write(f"OCR-like extracted text:\n{output_chunk[-1]}\n")
                elif output_format == 'csv':
                    for i, line in enumerate(output_chunk[-1].splitlines(), start=1):
                        output_file_obj.write(json.dumps({"page": i, "content": line}) + "\n")
                elif output_format == 'json':
                    json.dump({"content": output_chunk[-1]}, output_file_obj, indent=4)
                    output_file_obj.write("\n")
                elif output_format == 'jsonl':
                    for line in output_chunk[-1].splitlines():
                        json.dump({"content": line}, output_file_obj)
                        output_file_obj.write("\n")

            print(f"Written chunk {chunk_counter} for {pdf_file}")
            chunk_counter += 1

            # Compress chunk if necessary
            compress_file(output_file, compression_type)

        print(f"Successfully processed: {pdf_file}")

    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        log_error(pdf_file, str(e))
        continue  # Move to the next file

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
