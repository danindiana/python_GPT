
# PDF OCR Processing Script

This repository contains a Python script for processing PDF files to extract text using OCR (Optical Character Recognition) and save the output in various formats (txt, csv, json). The script leverages GPU for efficient processing and includes error logging and progress tracking.

## Features

- **OCR Text Extraction**: Extracts text from PDF files using OCR.
- **Multiple Output Formats**: Supports saving the extracted text in txt, csv, and json formats.
- **GPU Utilization**: Allows selection of GPU for processing to enhance performance.
- **Progress Tracking**: Keeps track of processed files to resume processing if interrupted.
- **Error Logging**: Logs errors encountered during processing for debugging.

## Requirements

- Python 3.x
- PyTorch
- Tesseract OCR
- Other dependencies listed in the script

## Directory Structure

```
.
├── combined_output.json
├── combined_output.py
├── combined_output.txt
├── error.log
├── error_logger.py
├── error_log.py
├── gpu_selection.py
├── main_script.py
├── main_scriptv2.py
├── main_scriptv3xp.py
├── model_utils.py
├── pdf_ocr_utils.py
├── preprocessing.py
├── processed_files.log
├── progress_tracker.py
├── __pycache__
│   ├── error_logger.cpython-312.pyc
│   ├── gpu_selection.cpython-312.pyc
│   ├── model_utils.cpython-312.pyc
│   ├── pdf_ocr_utils.cpython-312.pyc
│   ├── preprocessing.cpython-312.pyc
│   ├── progress_tracker.cpython-312.pyc
│   └── pymupdf_utils.cpython-312.pyc
└── pymupdf_utils.py
```

## Usage

1. **Set TESSDATA_PREFIX**: Ensure the `TESSDATA_PREFIX` environment variable is set correctly to the path where Tesseract OCR data is stored.
2. **Input and Output Directories**: Provide the paths for the input directory containing PDF files and the output directory for processed text files.
3. **Output Format**: Choose the desired output format (txt, csv, json).
4. **GPU Selection**: Select the GPU to be used for processing.
5. **Run the Script**: Execute the script to process the PDF files.

## Example

```bash
python main_script.py
```

## Script Details

### Environment Setup

```python
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
```

### User Input

```python
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
```

### Progress Tracking

```python
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
```

### GPU Selection

```python
# Fetch GPU information and prompt user to select a GPU
gpus = get_gpu_info()
selected_gpu = select_gpu(gpus)
torch.cuda.set_device(selected_gpu)

# Load model and processor only after directory confirmation to delay GPU allocation
device = torch.device(f"cuda:{selected_gpu}")
model, processor = load_model_and_processor(device)
```

### Processing PDF Files

```python
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
```

### Memory Cleanup

```python
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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
