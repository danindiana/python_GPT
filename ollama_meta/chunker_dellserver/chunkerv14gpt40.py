import os
import requests
import datetime
import json
import logging
import time
import signal
import subprocess
import argparse
from tqdm import tqdm

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # Corrected endpoint
MODEL_NAME = "qwen2.5-coder:3b"  # Updated model name
CHUNK_SIZE = 1024  # Characters per chunk (adjust as needed)
DEFAULT_OUTPUT_BASE = "/home/spook/processed_output"  # Updated base directory for default output
API_REQUEST_DELAY = 1  # Seconds to delay between API requests to avoid rate limiting

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Signal handling for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info(f"Signal {sig} received. Preparing for shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_text_with_ollama(prompt):
    """Send a prompt to the Ollama API and return the response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,  # Adjust temperature as needed
        "max_tokens": 100  # Adjust max tokens as needed
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)  # Added timeout
        response.raise_for_status()  # Raise HTTP errors if any

        # Log full raw response
        logging.debug(f"Raw API response: {response.text}")
        with open("api_responses.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"Prompt: {prompt[:50]}...\nResponse: {response.text}\n\n")

        # If response is NDJSON (newline-delimited JSON)
        responses = []
        for line in response.text.splitlines():
            try:
                response_data = json.loads(line)
                responses.append(response_data.get("response", "").strip())
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing line: {line}\nError: {e}")

        return "\n".join(responses)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Ollama: {e}")
        return f"Error: Connection issue - {e}"

def describe_document_with_ollama(file_name, text):
    """Send a request to the Ollama API to describe the document and extract details."""
    prompt = f"""Please analyze the following document and provide:
    1. A brief summary.
    2. Key topics covered.
    3. The type of document (e.g., technical report, essay, legal document, etc.).
    4. Any other relevant metadata or observations.

    Document Name: {file_name}
    Content: {text[:1000]}... (truncated for analysis if too long)
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 500  # Increase token limit for detailed descriptions
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)  # Added timeout
        response.raise_for_status()

        logging.debug(f"Raw API response for description: {response.text}")

        # Parse JSON response
        response_data = response.json()
        return response_data.get("response", "").strip()

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        logging.debug(f"Raw response content: {response.text}")
        return f"Error: Unable to parse description. Check server output."

    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to Ollama for description: {e}")
        return f"Error: Connection issue - {e}"

def chunk_text(text, chunk_size):
    """Split text into chunks of a specified size."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def process_files(input_dir, output_dir):
    """Iterate over text files in input_dir, process them, and save outputs."""
    for filename in tqdm(os.listdir(input_dir), desc="Processing files"):
        if filename.endswith(".txt"):  # Process only .txt files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as infile:
                text = infile.read()

            # Step 1: Generate document description
            logging.info(f"\nDescribing document: {filename}...")
            description = describe_document_with_ollama(filename, text)
            logging.info(f"Document description:\n{description}\n")

            # Step 2: Process the file in chunks if needed
            processed_text = []
            for chunk in chunk_text(text, CHUNK_SIZE):
                if shutdown_requested:
                    logging.info("Shutdown requested. Stopping processing...")
                    return
                logging.info(f"Processing chunk from {filename}...")
                processed_chunk = process_text_with_ollama(chunk)
                logging.debug(f"Processed chunk: {processed_chunk}")  # Print the processed text
                if processed_chunk:  # Only append non-empty responses
                    processed_text.append(processed_chunk)
                time.sleep(API_REQUEST_DELAY)  # Delay to avoid rate limiting

            # Save the processed output
            if processed_text:  # Only save if there's processed text
                final_output = "\n".join(processed_text)
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.write(final_output)
                logging.info(f"Processed file saved: {output_path}")
                logging.debug(f"Final content written to {output_path}:\n{final_output}")
            else:
                logging.warning(f"No processed text for {filename}. Skipping save.")

def check_permissions(directory, permission_type):
    """Check if the program has the required permissions for a directory."""
    if permission_type == "read":
        return os.access(directory, os.R_OK)
    elif permission_type == "write":
        return os.access(directory, os.W_OK)
    return False

def ensure_output_directory(output_dir):
    """Ensure the output directory exists and has write permissions."""
    try:
        if not os.path.exists(output_dir):
            logging.info(f"Output directory '{output_dir}' does not exist. Creating it...")
            os.makedirs(output_dir, exist_ok=True)

        if not check_permissions(output_dir, "write"):
            logging.error(f"Error: Permission denied for writing to {output_dir}. Exiting.")
            return False

        logging.info(f"Output directory '{output_dir}' is ready for use.")
        return True

    except Exception as e:
        logging.error(f"Error creating or accessing output directory: {e}")
        return False

def list_gpus():
    """List available GPUs and their utilization using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Error running nvidia-smi: {result.stderr}")
            return []

        gpus = result.stdout.strip().split('\n')
        gpu_info = []
        for gpu in gpus:
            index, name, utilization = gpu.split(', ')
            gpu_info.append((index, name, utilization))

        return gpu_info

    except Exception as e:
        logging.error(f"Error listing GPUs: {e}")
        return []

def select_gpu():
    """Prompt user to select a GPU."""
    gpus = list_gpus()
    if not gpus:
        logging.error("No GPUs found or unable to list GPUs.")
        return None

    print("Available GPUs:")
    for i, (index, name, utilization) in enumerate(gpus):
        print(f"{i + 1}. GPU {index}: {name} (Utilization: {utilization}%)")

    choice = input("Select a GPU by entering the corresponding number: ").strip()
    try:
        choice = int(choice)
        if 1 <= choice <= len(gpus):
            selected_gpu = gpus[choice - 1]
            logging.info(f"Selected GPU {selected_gpu[0]}: {selected_gpu[1]} (Utilization: {selected_gpu[2]}%)")
            return selected_gpu[0]
        else:
            logging.error("Invalid selection. Please try again.")
            return None
    except ValueError:
        logging.error("Invalid input. Please enter a number.")
        return None

def main():
    global shutdown_requested

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Process text files using Ollama API.")
    parser.add_argument("input_dir", help="Source directory containing text files.")
    parser.add_argument("--output_dir", help="Output directory for processed files. Default is a timestamped directory in DEFAULT_OUTPUT_BASE.")
    parser.add_argument("--gpu", help="Specify GPU index to use. Default is to prompt for selection.", type=int)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    specified_gpu = args.gpu

    if not os.path.isdir(input_dir):
        logging.error(f"Error: {input_dir} is not a valid directory. Exiting.")
        return

    if not check_permissions(input_dir, "read"):
        logging.error(f"Error: Permission denied for reading files in {input_dir}. Exiting.")
        return

    # Step 2: Select GPU
    if specified_gpu is not None:
        selected_gpu = specified_gpu
        logging.info(f"Using specified GPU {selected_gpu}.")
    else:
        selected_gpu = select_gpu()
        if not selected_gpu:
            logging.error("GPU selection failed. Exiting.")
            return

    # Step 3: Set up output directory
    if not output_dir:
        default_output_dir = os.path.join(DEFAULT_OUTPUT_BASE, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        output_dir = default_output_dir

    if not ensure_output_directory(output_dir):
        return

    # Step 4: Process files
    logging.info(f"\nProcessing files in {input_dir}...")
    process_files(input_dir, output_dir)
    logging.info(f"\nProcessing complete! Outputs saved to {output_dir}")

    # Step 5: Handle graceful shutdown
    if shutdown_requested:
        logging.info("Graceful shutdown initiated...")
        # Add any cleanup code here if needed

if __name__ == "__main__":
    main()
