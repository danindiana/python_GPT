The error message `Error: unknown flag: --port` indicates that the `ollama run` command does not accept the `--port` flag. This suggests that the `ollama run` command does not support specifying the port directly.

To address this, we need to find another way to ensure that the `ollama run` command uses the correct port. One approach is to set the `OLLAMA_HOST` environment variable before running the `ollama run` command. This way, the `ollama` command will use the specified host and port.

Here’s the updated script:

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

def main(source_dir, output_dir, chunk_size):
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
    if len(sys.argv) != 4:
        print("Usage: python chunksduck.py <source_dir> <output_dir> <chunk_size>")
        sys.exit(1)

    # Arguments: source directory, output directory, and chunk size
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    chunk_size = sys.argv[3]

    main(source_dir, output_dir, chunk_size)
```

### Explanation:
1. **Fetch the Running Model and Port**: The `get_running_model_and_port` function runs `ollama ps` and parses the output to get the currently running model's name and the port. It now checks if there are at least 5 elements in the split line to ensure the model name and port are correctly parsed.
2. **Set the `OLLAMA_HOST` Environment Variable**: The `process_chunk` function sets the `OLLAMA_HOST` environment variable to the specified host and port before running the `ollama run` command.
3. **Error Handling**: If no running model or port is found, the script prints an error message and exits.

### Usage:
```bash
python chunksduck.py /home/programs/target-dir /home/programs/destination-dir 5000
```

This command will process all `.txt` files in `/home/your/target-dir`, chunk them into pieces of 5000 words each, process each chunk with the currently running `ollama` model and port, and save the results in `/home/smduck/programs/temp_txt_llm_postprocess`. If `ollama` encounters an error, it will print the error message and continue processing the next chunk.
