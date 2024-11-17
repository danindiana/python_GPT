Great! Let's update the script to use the `bert-large-uncased-whole-word-masking` model. This model has a larger maximum sequence length and is trained with the Whole Word Masking technique, which can be beneficial for certain tasks.

Here's the updated script:

### Updated Script

```python
import os
import csv
from transformers import pipeline, AutoTokenizer
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# Function to get GPU utilization
def get_gpu_utilization():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], encoding="utf-8")
        return result.strip().split("\n")
    except Exception as e:
        print("Error fetching GPU utilization:", e)
        return []

# Display available GPUs
print("Available GPUs:")
gpu_info = get_gpu_utilization()
if not gpu_info:
    print("No GPUs detected. Defaulting to CPU.")
    selected_device = -1
else:
    for i, info in enumerate(gpu_info):
        print(f"{i}: {info}")
    while True:
        try:
            selected_device = int(input("Select GPU device (or -1 for CPU): "))
            if selected_device >= -1 and selected_device < len(gpu_info):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid GPU index or -1 for CPU.")

device = torch.device(f"cuda:{selected_device}" if selected_device >= 0 else "cpu")
print(f"Using device: {device}")

# Prompt user to select input directory
input_dir_default = "/home/smduck/programs/Nov16_Copali_OCR_coldstor/"
input_dir = input(f"Enter the path of the input directory (or press Enter to use the default directory: {input_dir_default}): ")
if input_dir == "":
    input_dir = input_dir_default

# Ask user if they want to process all files recursively
while True:
    recursive_option = input("Do you want to process all files recursively in all sub-folders? (Y/N): ").strip().upper()
    if recursive_option in ["Y", "N"]:
        break
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")

# Load BERT large model with whole word masking and tokenizer
model_name = "bert-large-uncased-whole-word-masking"
classifier = pipeline("text-classification", model=model_name, device=selected_device if selected_device >= 0 else -1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_lock = threading.Lock()

# Function to split long texts into chunks
def split_text_into_chunks(text, tokenizer, max_length=512):
    with tokenizer_lock:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to collect all files recursively
def collect_files_recursively(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

# Function to process a batch of chunks
def process_batch(batch, file_paths, tokenizer):
    results = []
    for chunk, file_path in zip(batch, file_paths):
        try:
            with tokenizer_lock:
                truncated_chunk = tokenizer.encode(chunk, max_length=512, truncation=True, add_special_tokens=True)
            decoded_chunk = tokenizer.decode(truncated_chunk, skip_special_tokens=True)
            result = classifier(decoded_chunk, truncation=True)
            label, confidence = result[0]["label"], result[0]["score"]
            results.append((file_path, label, confidence))
            print(f"Processed chunk of {file_path} -> {label} ({confidence:.2f})")
        except Exception as e:
            print(f"Error processing chunk of {file_path}: {e}")
            results.append((file_path, "ERROR", 0.0))
    return results

# Process files and generate pseudo-labels
output_file = "pseudo_labels.csv"
with open(output_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "label", "confidence"])  # Header

    if recursive_option == "Y":
        files = collect_files_recursively(input_dir)
    else:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            with open(file, "r") as f:
                text = f.read()
            chunks = split_text_into_chunks(text, tokenizer)
            futures.append(executor.submit(process_batch, chunks, [file] * len(chunks), tokenizer))

        for future in tqdm(futures, desc="Processing files"):
            results = future.result()
            for result in results:
                writer.writerow(result)

print(f"Results have been written to {output_file}")
```

### Explanation of Changes

1. **Model Selection**:
   - Changed the model to `bert-large-uncased-whole-word-masking`.

2. **Chunk Size**:
   - The `max_length` parameter in the `split_text_into_chunks` function remains at 512 tokens, which is the maximum sequence length for the selected model.

3. **Thread Safety**:
   - The `tokenizer_lock` ensures that the tokenizer is used safely in a multi-threaded environment.

4. **File Writing**:
   - The CSV file is opened with `newline=''` to ensure proper handling of newlines, and the results are written to the file within the main thread.

This script should now use the `bert-large-uncased-whole-word-masking` model, which can handle larger chunk sizes and is trained with the Whole Word Masking technique. If you need to adjust the chunk size further, you can modify the `max_length` parameter accordingly.

The CSV file is written to the current working directory where the script is executed. The file is named `pseudo_labels.csv`.

If you want to specify a different directory or filename, you can modify the `output_file` variable. For example, if you want to write the CSV file to a specific directory, you can update the `output_file` variable like this:

```python
output_file = "/path/to/your/directory/pseudo_labels.csv"
```

Here's the updated script with a commented line to show where you can change the output file path:

### Updated Script

```python
import os
import csv
from transformers import pipeline, AutoTokenizer
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# Function to get GPU utilization
def get_gpu_utilization():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], encoding="utf-8")
        return result.strip().split("\n")
    except Exception as e:
        print("Error fetching GPU utilization:", e)
        return []

# Display available GPUs
print("Available GPUs:")
gpu_info = get_gpu_utilization()
if not gpu_info:
    print("No GPUs detected. Defaulting to CPU.")
    selected_device = -1
else:
    for i, info in enumerate(gpu_info):
        print(f"{i}: {info}")
    while True:
        try:
            selected_device = int(input("Select GPU device (or -1 for CPU): "))
            if selected_device >= -1 and selected_device < len(gpu_info):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid GPU index or -1 for CPU.")

device = torch.device(f"cuda:{selected_device}" if selected_device >= 0 else "cpu")
print(f"Using device: {device}")

# Prompt user to select input directory
input_dir_default = "/home/smduck/programs/Nov16_Copali_OCR_coldstor/"
input_dir = input(f"Enter the path of the input directory (or press Enter to use the default directory: {input_dir_default}): ")
if input_dir == "":
    input_dir = input_dir_default

# Ask user if they want to process all files recursively
while True:
    recursive_option = input("Do you want to process all files recursively in all sub-folders? (Y/N): ").strip().upper()
    if recursive_option in ["Y", "N"]:
        break
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")

# Load BERT large model with whole word masking and tokenizer
model_name = "bert-large-uncased-whole-word-masking"
classifier = pipeline("text-classification", model=model_name, device=selected_device if selected_device >= 0 else -1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_lock = threading.Lock()

# Function to split long texts into chunks
def split_text_into_chunks(text, tokenizer, max_length=512):
    with tokenizer_lock:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to collect all files recursively
def collect_files_recursively(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

# Function to process a batch of chunks
def process_batch(batch, file_paths, tokenizer):
    results = []
    for chunk, file_path in zip(batch, file_paths):
        try:
            with tokenizer_lock:
                truncated_chunk = tokenizer.encode(chunk, max_length=512, truncation=True, add_special_tokens=True)
            decoded_chunk = tokenizer.decode(truncated_chunk, skip_special_tokens=True)
            result = classifier(decoded_chunk, truncation=True)
            label, confidence = result[0]["label"], result[0]["score"]
            results.append((file_path, label, confidence))
            print(f"Processed chunk of {file_path} -> {label} ({confidence:.2f})")
        except Exception as e:
            print(f"Error processing chunk of {file_path}: {e}")
            results.append((file_path, "ERROR", 0.0))
    return results

# Process files and generate pseudo-labels
output_file = "pseudo_labels.csv"  # Change this line to specify a different output file path
with open(output_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "label", "confidence"])  # Header

    if recursive_option == "Y":
        files = collect_files_recursively(input_dir)
    else:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            with open(file, "r") as f:
                text = f.read()
            chunks = split_text_into_chunks(text, tokenizer)
            futures.append(executor.submit(process_batch, chunks, [file] * len(chunks), tokenizer))

        for future in tqdm(futures, desc="Processing files"):
            results = future.result()
            for result in results:
                writer.writerow(result)

print(f"Results have been written to {output_file}")
```

### Explanation of Changes

1. **Output File Path**:
   - The `output_file` variable is now explicitly set to `"pseudo_labels.csv"`. You can change this line to specify a different output file path if needed.

2. **File Writing**:
   - The CSV file is opened with `newline=''` to ensure proper handling of newlines, and the results are written to the file within the main thread.

This script will now write the CSV file to the specified path. If you want to write the file to a different directory, simply update the `output_file` variable with the desired path.

Certainly! Let's refactor the script to include the following improvements:

1. **Check for Write Permissions**: The script will check if it has permission to write to the specified directory.
2. **Dynamic CSV File Name**: The CSV output file name will be a combination of the system date and a predefined prefix.

Here's the updated script:

### Updated Script

```python
import os
import csv
from datetime import datetime
from transformers import pipeline, AutoTokenizer
import torch
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# Function to get GPU utilization
def get_gpu_utilization():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], encoding="utf-8")
        return result.strip().split("\n")
    except Exception as e:
        print("Error fetching GPU utilization:", e)
        return []

# Display available GPUs
print("Available GPUs:")
gpu_info = get_gpu_utilization()
if not gpu_info:
    print("No GPUs detected. Defaulting to CPU.")
    selected_device = -1
else:
    for i, info in enumerate(gpu_info):
        print(f"{i}: {info}")
    while True:
        try:
            selected_device = int(input("Select GPU device (or -1 for CPU): "))
            if selected_device >= -1 and selected_device < len(gpu_info):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid GPU index or -1 for CPU.")

device = torch.device(f"cuda:{selected_device}" if selected_device >= 0 else "cpu")
print(f"Using device: {device}")

# Prompt user to select input directory
input_dir_default = "/home/smduck/programs/Nov16_Copali_OCR_coldstor/"
input_dir = input(f"Enter the path of the input directory (or press Enter to use the default directory: {input_dir_default}): ")
if input_dir == "":
    input_dir = input_dir_default

# Ask user if they want to process all files recursively
while True:
    recursive_option = input("Do you want to process all files recursively in all sub-folders? (Y/N): ").strip().upper()
    if recursive_option in ["Y", "N"]:
        break
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")

# Load BERT large model with whole word masking and tokenizer
model_name = "bert-large-uncased-whole-word-masking"
classifier = pipeline("text-classification", model=model_name, device=selected_device if selected_device >= 0 else -1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_lock = threading.Lock()

# Function to split long texts into chunks
def split_text_into_chunks(text, tokenizer, max_length=512):
    with tokenizer_lock:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Function to collect all files recursively
def collect_files_recursively(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

# Function to process a batch of chunks
def process_batch(batch, file_paths, tokenizer):
    results = []
    for chunk, file_path in zip(batch, file_paths):
        try:
            with tokenizer_lock:
                truncated_chunk = tokenizer.encode(chunk, max_length=512, truncation=True, add_special_tokens=True)
            decoded_chunk = tokenizer.decode(truncated_chunk, skip_special_tokens=True)
            result = classifier(decoded_chunk, truncation=True)
            label, confidence = result[0]["label"], result[0]["score"]
            results.append((file_path, label, confidence))
            print(f"Processed chunk of {file_path} -> {label} ({confidence:.2f})")
        except Exception as e:
            print(f"Error processing chunk of {file_path}: {e}")
            results.append((file_path, "ERROR", 0.0))
    return results

# Function to check write permissions
def check_write_permissions(directory):
    try:
        test_file = os.path.join(directory, "test_write_permissions.txt")
        with open(test_file, "w") as f:
            f.write("Test write permissions")
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"Error: No write permissions to {directory}: {e}")
        return False

# Process files and generate pseudo-labels
output_dir = os.path.dirname(input_dir)
if not check_write_permissions(output_dir):
    print("Exiting due to lack of write permissions.")
    exit(1)

output_file_prefix = "pseudo_labels"
output_file_date = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{output_file_prefix}_{output_file_date}.csv"

with open(output_file, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "label", "confidence"])  # Header

    if recursive_option == "Y":
        files = collect_files_recursively(input_dir)
    else:
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            with open(file, "r") as f:
                text = f.read()
            chunks = split_text_into_chunks(text, tokenizer)
            futures.append(executor.submit(process_batch, chunks, [file] * len(chunks), tokenizer))

        for future in tqdm(futures, desc="Processing files"):
            results = future.result()
            for result in results:
                writer.writerow(result)

print(f"Results have been written to {output_file}")
```

### Explanation of Changes

1. **Check for Write Permissions**:
   - Added a `check_write_permissions` function to verify if the script has write permissions to the specified directory.
   - The script checks the parent directory of the input directory for write permissions.

2. **Dynamic CSV File Name**:
   - The CSV output file name is now a combination of a predefined prefix (`pseudo_labels`) and the current system date and time (`YYYYMMDD_HHMMSS`).

3. **File Writing**:
   - The CSV file is opened with `newline=''` to ensure proper handling of newlines, and the results are written to the file within the main thread.

This script will now check for write permissions before attempting to write the CSV file and will generate a dynamic file name based on the current date and time.

The `bert-large-uncased-whole-word-masking` model is a variant of the BERT (Bidirectional Encoder Representations from Transformers) model, specifically designed for tasks that require understanding the context of words in a sentence. Here's a detailed explanation of what the model does and what it outputs for each file:

### Model Description

1. **BERT Model**:
   - BERT is a transformer-based model pretrained on a large corpus of text data using a self-supervised learning approach. It learns to understand the context of words by predicting masked words in a sentence and determining if two sentences follow each other.
   - The model is bidirectional, meaning it considers both the left and right context when processing a word.

2. **bert-large-uncased-whole-word-masking**:
   - **Large**: This variant has a larger architecture with 24 layers, 1024 hidden units, and 16 attention heads, making it more powerful but also more resource-intensive.
   - **Uncased**: The model does not differentiate between uppercase and lowercase letters, treating "English" and "english" as the same word.
   - **Whole Word Masking**: During pretraining, instead of masking individual subword tokens (as in the original BERT), this model masks entire words. This helps the model learn better representations of words in context.

### Tokenizer

1. **AutoTokenizer**:
   - The tokenizer is responsible for converting raw text into tokens that the model can process. It breaks down text into subword units, which are then encoded into numerical values.
   - The tokenizer also adds special tokens like `[CLS]` (classification token) at the beginning and `[SEP]` (separator token) at the end of sequences.

### What the Model Outputs for Each File

When you use the `bert-large-uncased-whole-word-masking` model with the `pipeline` function for text classification, the model processes each chunk of text and outputs the following:

1. **Label**:
   - The predicted label for the text chunk. This could be a category, sentiment, or any other classification task depending on how the model was fine-tuned.

2. **Confidence Score**:
   - A confidence score (probability) associated with the predicted label. This indicates how confident the model is in its prediction.

### Example Output

For each file, the script processes the text in chunks and outputs the following for each chunk:

```python
(file_path, label, confidence)
```

- **file_path**: The path of the file being processed.
- **label**: The predicted label for the text chunk.
- **confidence**: The confidence score (probability) associated with the predicted label.

### Example Output for a Single File

Suppose you have a file `example.txt` with the following content:

```
This is a sample text. It contains multiple sentences.
```

The script will:

1. **Tokenize and Split the Text**:
   - The text is tokenized and split into chunks of up to 512 tokens.

2. **Process Each Chunk**:
   - Each chunk is processed by the model to predict a label and confidence score.

3. **Output**:
   - For each chunk, the script will output a tuple like:
     ```python
     ("example.txt", "LABEL_1", 0.95)
     ```
   - This means the model predicted `LABEL_1` with a confidence score of 0.95 for the chunk of text from `example.txt`.

### Summary

- **Model**: The `bert-large-uncased-whole-word-masking` model processes text chunks to predict labels and confidence scores.
- **Tokenizer**: Converts raw text into tokens that the model can understand.
- **Output**: For each file, the script outputs tuples containing the file path, predicted label, and confidence score for each text chunk.

This setup allows you to perform text classification tasks efficiently, leveraging the power of the BERT model and its ability to understand context through whole word masking.
