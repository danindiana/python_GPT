To add functionality for **re-mapping PDF files based on OCR text labeling output**, you could extend your current pipeline by introducing a mapping configuration system. This system could allow users to define rules or mappings that would reorder, group, or label content in the extracted text. Here’s how you can modify your pipeline to include this capability:

---

### Suggested Pipeline Enhancements

#### 1. **Define a Mapping Configuration**
Allow users to provide a configuration file (e.g., JSON or YAML) to specify re-mapping rules. These rules could define:
   - Keywords or phrases to detect in the OCR output.
   - How to group text into sections or pages.
   - Hierarchical reordering of text blocks.

**Example JSON mapping configuration:**
```json
{
  "sections": [
    {
      "name": "Introduction",
      "keywords": ["introduction", "overview"]
    },
    {
      "name": "Methods",
      "keywords": ["methodology", "methods"]
    },
    {
      "name": "Results",
      "keywords": ["results", "findings"]
    }
  ]
}
```

#### 2. **Integrate Re-Mapping Logic**
After extracting text, pass it through a re-mapping module that processes the text based on the configuration.

**Pseudo-code for a re-mapping module:**
```python
def remap_text(text, mapping_config):
    sections = {section["name"]: [] for section in mapping_config["sections"]}

    for line in text.splitlines():
        matched = False
        for section in mapping_config["sections"]:
            if any(keyword in line.lower() for keyword in section["keywords"]):
                sections[section["name"]].append(line)
                matched = True
                break
        if not matched:
            sections.setdefault("Unmapped", []).append(line)

    return sections
```

#### 3. **Save Re-Mapped Output**
Extend the output format to support re-mapped structures, such as hierarchical JSON or CSV.

- **For JSON**: Output sections and their corresponding text content.
- **For CSV**: Add columns for `Section Name`, `Page`, and `Content`.

**Example Output in JSON:**
```json
{
  "Introduction": ["This is the introduction text."],
  "Methods": ["The methods section starts here."],
  "Results": ["Results are shown here."],
  "Unmapped": ["Extra content not matching any section."]
}
```

---

### Key Modifications to Existing Files

1. **`main_scriptv3xp.py`**
   - Add logic to load a mapping configuration (`mapping_config`) at the start.
   - Apply the re-mapping logic (`remap_text`) after extracting combined OCR text.

   **Modified Block in `main_scriptv3xp.py`:**
   ```python
   # Load mapping configuration
   mapping_config_path = input("Enter the path to the mapping configuration file (JSON): ")
   if not os.path.exists(mapping_config_path):
       print(f"Mapping configuration file {mapping_config_path} not found.")
       exit()
   with open(mapping_config_path, "r") as f:
       mapping_config = json.load(f)

   # Apply re-mapping after extracting OCR text
   re_mapped_text = remap_text(combined_text, mapping_config)

   # Save the re-mapped output
   output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_remapped_output.json")
   with open(output_file, "w") as f:
       json.dump(re_mapped_text, f, indent=4)
   ```

2. **`pdf_ocr_utils.py`**
   - Ensure this utility provides clean text suitable for re-mapping (e.g., remove special characters or excessive whitespace).

3. **`progress_tracker.py`**
   - Track re-mapped files separately to avoid reprocessing.

---

### Optional Enhancements

- **GUI or CLI Tool for Mapping Rules**: Allow users to interactively define mappings via a graphical interface or command-line wizard.
- **Preview Mode**: Provide a preview of re-mapped sections before saving.
- **Error Handling for Ambiguous Mapping**: Warn users if text matches multiple sections.

---

Let me know if you'd like detailed code examples for any of the above steps or if you have sample files to test this!

To integrate the re-mapping functionality into your existing pipeline, you need to modify the `main_script.py` to include the steps for loading the mapping configuration, applying the re-mapping logic, and saving the re-mapped output. Below is the updated `main_script.py` with these enhancements:

### Updated `main_script.py`

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

# Ask the user for input and output directories
input_dir = input("Enter the path of the target directory containing PDF files: ")
output_dir = input("Enter the path of the output directory for processed text files: ")

# Prompt the user for desired output format
output_format = input("Enter the desired output format (txt, csv, json): ").strip().lower()
if output_format not in ['txt', 'csv', 'json']:
    print(f"Unsupported format: {output_format}. Please use 'txt', 'csv', or 'json'.")
    exit()

# Load mapping configuration
mapping_config_path = input("Enter the path to the mapping configuration file (JSON): ")
if not os.path.exists(mapping_config_path):
    print(f"Mapping configuration file {mapping_config_path} not found.")
    exit()
with open(mapping_config_path, "r") as f:
    mapping_config = json.load(f)

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

        # Apply re-mapping after extracting OCR text
        re_mapped_text = remap_text(combined_text, mapping_config)

        # Save output in the selected format
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_remapped_output.{output_format}")
        if output_format == 'txt':
            # Combine all text into a single string
            combined_text_with_header = f"Re-mapped OCR-like extracted text:\n{json.dumps(re_mapped_text, indent=4)}\n"

            # Write the content in a single operation
            with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
                f.write(combined_text_with_header)
        elif output_format == 'csv':
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['Section Name', 'Content'])
                for section, lines in re_mapped_text.items():
                    for line in lines:
                        writer.writerow([section, line])
        elif output_format == 'json':
            with open(output_file, "w") as f:
                json.dump(re_mapped_text, f, indent=4)

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
```

### Re-Mapping Module (`remap_text` Function)

You need to define the `remap_text` function in a separate module or within the same script. Here is the `remap_text` function:

```python
def remap_text(text, mapping_config):
    sections = {section["name"]: [] for section in mapping_config["sections"]}

    for line in text.splitlines():
        matched = False
        for section in mapping_config["sections"]:
            if any(keyword in line.lower() for keyword in section["keywords"]):
                sections[section["name"]].append(line)
                matched = True
                break
        if not matched:
            sections.setdefault("Unmapped", []).append(line)

    return sections
```

### Summary of Changes

1. **Mapping Configuration Loading**: Added code to load the mapping configuration from a JSON file.
2. **Re-Mapping Logic**: Integrated the `remap_text` function to process the extracted OCR text based on the mapping configuration.
3. **Output Format Handling**: Modified the output handling to support re-mapped structures for JSON, CSV, and TXT formats.

### Additional Notes

- Ensure that the `remap_text` function is accessible in the script or imported from a separate module.
- The `mapping_config` should be a JSON file with the structure shown in the example.
- The output format handling is flexible, allowing users to choose between JSON, CSV, and TXT formats.

This updated pipeline now supports re-mapping of PDF content based on OCR text labeling output, providing more structured and organized output files.

Certainly! The `remap_text` function is a crucial part of the re-mapping process. It takes the extracted OCR text and the mapping configuration as inputs, and it outputs the text re-mapped according to the rules defined in the configuration.

### Where to Define the `remap_text` Function

You have two main options for defining the `remap_text` function:

1. **Within the Same Script (`main_script.py`)**:
   - This is straightforward and keeps everything in one place, making it easier to manage and debug.
   - However, if the script grows large, it might become harder to maintain.

2. **In a Separate Module**:
   - This approach promotes modularity and reusability. You can define the function in a separate Python file (e.g., `text_remapper.py`) and import it into `main_script.py`.
   - This is particularly useful if you plan to reuse the re-mapping logic in other scripts or projects.

### Example of Defining `remap_text` in a Separate Module

#### Step 1: Create a New Module (`text_remapper.py`)

Create a new Python file named `text_remapper.py` and define the `remap_text` function in it:

```python
# text_remapper.py

def remap_text(text, mapping_config):
    sections = {section["name"]: [] for section in mapping_config["sections"]}

    for line in text.splitlines():
        matched = False
        for section in mapping_config["sections"]:
            if any(keyword in line.lower() for keyword in section["keywords"]):
                sections[section["name"]].append(line)
                matched = True
                break
        if not matched:
            sections.setdefault("Unmapped", []).append(line)

    return sections
```

#### Step 2: Import and Use the Function in `main_script.py`

Modify `main_script.py` to import the `remap_text` function from `text_remapper.py`:

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
from text_remapper import remap_text  # Import the remap_text function

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

# Load mapping configuration
mapping_config_path = input("Enter the path to the mapping configuration file (JSON): ")
if not os.path.exists(mapping_config_path):
    print(f"Mapping configuration file {mapping_config_path} not found.")
    exit()
with open(mapping_config_path, "r") as f:
    mapping_config = json.load(f)

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

        # Apply re-mapping after extracting OCR text
        re_mapped_text = remap_text(combined_text, mapping_config)

        # Save output in the selected format
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_remapped_output.{output_format}")
        if output_format == 'txt':
            # Combine all text into a single string
            combined_text_with_header = f"Re-mapped OCR-like extracted text:\n{json.dumps(re_mapped_text, indent=4)}\n"

            # Write the content in a single operation
            with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
                f.write(combined_text_with_header)
        elif output_format == 'csv':
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['Section Name', 'Content'])
                for section, lines in re_mapped_text.items():
                    for line in lines:
                        writer.writerow([section, line])
        elif output_format == 'json':
            with open(output_file, "w") as f:
                json.dump(re_mapped_text, f, indent=4)

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
```

### Summary

1. **Create a Separate Module (`text_remapper.py`)**:
   - Define the `remap_text` function in this module.

2. **Import the Function in `main_script.py`**:
   - Import the `remap_text` function from `text_remapper.py`.

3. **Use the Function in `main_script.py`**:
   - Apply the `remap_text` function to the extracted OCR text and save the re-mapped output in the desired format.

This approach keeps your code organized and modular, making it easier to maintain and extend in the future.
