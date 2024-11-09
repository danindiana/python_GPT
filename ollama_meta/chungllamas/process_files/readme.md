```
# process_files.sh
#!/bin/bash

# Define directories
TARGET_DIR="your/home/files-2b-processed"
OUTPUT_DIR="output/dir/files"
CHUNK_SIZE=4000  # Number of words per chunk, adjust as needed
CONTEXT_WINDOW=8096  # Set your desired context window size

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over each .txt file in target directory
find "$TARGET_DIR" -type f -name "*.txt" | while read -r FILE_PATH; do
    echo "Processing $FILE_PATH"

    # Call the Python script with context window
    python3 chunk_process.py "$FILE_PATH" "$OUTPUT_DIR" "$CHUNK_SIZE" "$CONTEXT_WINDOW"
done
```

### Directory Structure
```
.
├── process_files.sh
└── chunk_process.py
```

### `process_files.sh`
```bash
#!/bin/bash

# Define directories
TARGET_DIR="your/home/files-2b-processed"
OUTPUT_DIR="output/dir/files"
CHUNK_SIZE=4000  # Number of words per chunk, adjust as needed
CONTEXT_WINDOW=8096  # Set your desired context window size

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over each .txt file in target directory
find "$TARGET_DIR" -type f -name "*.txt" | while read -r FILE_PATH; do
    echo "Processing $FILE_PATH"

    # Call the Python script with context window
    python3 chunk_process.py "$FILE_PATH" "$OUTPUT_DIR" "$CHUNK_SIZE" "$CONTEXT_WINDOW"
done
```

### `chunk_process.py`
```python
import os
import sys
from subprocess import run, PIPE, CalledProcessError

def get_running_model_and_port():
    # Run ollama ps to get the currently running model and port
    process = run(["ollama", "ps"], stdout=PIPE, stderr=PIPE, text=True, check=True)
    output = process.stdout
    
    # Parse the output to get the model name and port
    lines = output.strip().split('\n')
    if len(lines) > 1:
        model_info = lines[1].split()
        if len(model_info) >= 5:
            model_name = model_info[0]
            port = model_info[-1]  # Assuming the port is the last element in the line
            return model_name, port
    return None, None

def chunk_file(file_path, chunk_size):
    with open(file_path, 'r') as file:
        words = file.read().split()

    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]

def process_chunk(chunk_text, model, host):
    try:
        # Set the OLLAMA_HOST environment variable
        os.environ['OLLAMA_HOST'] = host
        
        # Run ollama with the specified model and pass the chunk as input
        process = run(["ollama", "run", model], input=chunk_text, text=True, stdout=PIPE, stderr=PIPE, check=True)
        return process.stdout
    except CalledProcessError as e:
        print(f"Error processing chunk: {e.stderr}")
        return None

def main(source_dir, output_dir, chunk_size, context_window):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the currently running model and port
    model, port = get_running_model_and_port()
    if model is None or port is None:
        print("No running model or port found. Please start a model using 'ollama serve'.")
        sys.exit(1)

    # Construct the host string
    host = f"127.0.0.1:{port}"

    # Iterate over all files in the source directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(source_dir, file_name)
            file_base_name = os.path.splitext(file_name)[0]

            # Chunk the file
            chunks = chunk_file(file_path, int(chunk_size))

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} for {file_base_name}")

                # Process the chunk with ollama
                result = process_chunk(chunk, model, host)

                if result is not None:
                    # Save result to output directory
                    output_file = os.path.join(output_dir, f"{file_base_name}_chunk_{i+1}.txt")
                    with open(output_file, 'w') as f:
                        f.write(result)

            print(f"Completed processing for {file_base_name}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python chunk_process.py <source_dir> <output_dir> <chunk_size> <context_window>")
        sys.exit(1)

    # Arguments: source directory, output directory, chunk size, and context window
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    chunk_size = sys.argv[3]
    context_window = sys.argv[4]

    main(source_dir, output_dir, chunk_size, context_window)
```

### Summary
- **`process_files.sh`**: A bash script that iterates over `.txt` files in a specified directory, calls a Python script to process each file, and saves the results to an output directory.
- **`chunk_process.py`**: A Python script that splits a text file into chunks, processes each chunk using the `ollama` command-line tool, and saves the results.

This structure ensures that the bash script and Python script are clearly separated, making it easier to manage and understand the workflow.
