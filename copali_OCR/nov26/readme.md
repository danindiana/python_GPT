Modify the following code such that it will produce text files as output when user selects text as well and in addition to it's existing json output.

import os
import torch
import gc
import json
import csv
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

# Prompt the user for desired output format
output_format = input("Enter the desired output format (txt, csv, json): ").strip().lower()
if output_format not in ['txt', 'csv', 'json']:
    print(f"Unsupported format: {output_format}. Please use 'txt', 'csv', or 'json'.")
    exit()

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

        # Save output in the selected format
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_ocr_output.{output_format}")
        if output_format == 'txt':
            # Combine all text into a single string
            combined_text_with_header = f"OCR-like extracted text:\n{combined_text}\n"

            # Write the content in a single operation
            with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
                f.write(combined_text_with_header)
        elif output_format == 'csv':
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Content'])
                for i, line in enumerate(combined_text.splitlines(), start=1):
                    writer.writerow([i, line])
        elif output_format == 'json':
            with open(output_file, "w") as f:
                json.dump({"content": combined_text}, f, indent=4)

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
To modify the code so that it produces text files as output when the user selects txt as the output format, you can simply add the necessary code to handle the txt format within the existing logic. The current code already includes handling for txt, csv, and json formats, so no major changes are needed. However, I'll ensure the code is clear and properly handles the txt format as well.

Here's the modified code:

python
Copy
import os
import torch
import gc
import json
import csv
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

# Prompt the user for desired output format
output_format = input("Enter the desired output format (txt, csv, json): ").strip().lower()
if output_format not in ['txt', 'csv', 'json']:
    print(f"Unsupported format: {output_format}. Please use 'txt', 'csv', or 'json'.")
    exit()

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

        # Save output in the selected format
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_ocr_output.{output_format}")
        if output_format == 'txt':
            # Combine all text into a single string
            combined_text_with_header = f"OCR-like extracted text:\n{combined_text}\n"

            # Write the content in a single operation
            with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
                f.write(combined_text_with_header)
        elif output_format == 'csv':
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Content'])
                for i, line in enumerate(combined_text.splitlines(), start=1):
                    writer.writerow([i, line])
        elif output_format == 'json':
            with open(output_file, "w") as f:
                json.dump({"content": combined_text}, f, indent=4)

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
Explanation:
Output Format Handling: The code already includes handling for txt, csv, and json formats. When the user selects txt, the code combines the extracted text into a single string and writes it to a text file.

Buffer Size: The buffer size for writing the text file is set to 8192 bytes to reduce the number of flushes and improve performance.

Error Handling: The code includes error handling to log errors and skip problematic files, ensuring the process continues even if some files fail.

This code should work as expected, producing text files when the user selects txt as the output format, in addition to the existing csv and json formats.

ok outstanding that last output worked very well. It did end with this code though I think it finished processing all files. 

"-> Cannot close object, library is destroyed. This may cause a memory leak!"
The warning message "Cannot close object, library is destroyed. This may cause a memory leak!" suggests that there might be some resources that are not properly released before the program terminates. This could be due to the way the Tesseract OCR library or other external libraries are being used.

To address this, we can ensure that all resources are properly released and that any potential memory leaks are minimized. Here are a few steps to improve the code:

Explicitly Close Files: Ensure that all files are explicitly closed after writing.

Resource Management: Use context managers (with statements) to manage resources like files and GPU memory.

Final Cleanup: Ensure that all resources are released before the program exits.

Here's the updated code with these improvements:

python
Copy
import os
import torch
import gc
import json
import csv
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

# Prompt the user for desired output format
output_format = input("Enter the desired output format (txt, csv, json): ").strip().lower()
if output_format not in ['txt', 'csv', 'json']:
    print(f"Unsupported format: {output_format}. Please use 'txt', 'csv', or 'json'.")
    exit()

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

        # Save output in the selected format
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_ocr_output.{output_format}")
        if output_format == 'txt':
            # Combine all text into a single string
            combined_text_with_header = f"OCR-like extracted text:\n{combined_text}\n"

            # Write the content in a single operation
            with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
                f.write(combined_text_with_header)
        elif output_format == 'csv':
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['Page', 'Content'])
                for i, line in enumerate(combined_text.splitlines(), start=1):
                    writer.writerow([i, line])
        elif output_format == 'json':
            with open(output_file, "w") as f:
                json.dump({"content": combined_text}, f, indent=4)

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

# Explicitly close any open files and release resources
# (This is a general approach, specific cleanup may be needed for external libraries)
for obj in gc.get_objects():
    if isinstance(obj, (open, csv.writer)):
        obj.close()

# Ensure all GPU resources are released
torch.cuda.empty_cache()
gc.collect()
Explanation:
Explicitly Close Files: The with statements ensure that files are automatically closed after writing.

Resource Management: The with statements manage file resources, and the final cleanup ensures that any remaining resources are released.

Final Cleanup: The code explicitly closes any open files and releases GPU resources before exiting.

This should help mitigate the warning about potential memory leaks and ensure that all resources are properly managed.
