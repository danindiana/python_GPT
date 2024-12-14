import os
import requests
import datetime
import json
import logging
import time
import signal
import subprocess

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:3b"
CHUNK_SIZE = 1024  # Characters per chunk
DEFAULT_OUTPUT_BASE = "/home/spook/processed_output"
API_REQUEST_DELAY = 1  # Seconds between API requests

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Signal handling
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info(f"Signal {sig} received. Shutting down...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def send_api_request(payload, url=OLLAMA_URL):
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None

def process_text_with_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100
    }
    response_text = send_api_request(payload)
    if response_text:
        responses = []
        for line in response_text.splitlines():
            try:
                data = json.loads(line)
                responses.append(data.get("response", "").strip())
            except json.JSONDecodeError:
                pass
        return "\n".join(responses)
    return None

def describe_document_with_ollama(file_name, text):
    prompt = (
        f"Analyze the document '{file_name}' and provide a brief summary, key topics, document type, and other metadata.\n"
        f"Content: {text[:1000]}..."
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 500
    }
    response_text = send_api_request(payload)
    if response_text:
        try:
            response_data = json.loads(response_text)
            return response_data.get("response", "").strip()
        except json.JSONDecodeError:
            logging.error("Invalid JSON response from API.")
    return None

def chunk_text(text, chunk_size):
    return (text[i:i + chunk_size] for i in range(0, len(text), chunk_size))

def process_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"):
            continue
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        try:
            with open(input_path, "r", encoding="utf-8") as infile:
                text = infile.read()
        except Exception as e:
            logging.error(f"Error reading file {input_path}: {e}")
            continue

        logging.info(f"Describing document: {filename}")
        description = describe_document_with_ollama(filename, text)
        logging.info(f"Document description:\n{description}\n")

        processed_text = []
        for chunk in chunk_text(text, CHUNK_SIZE):
            if shutdown_requested:
                logging.info("Shutdown requested. Stopping processing.")
                return
            logging.info(f"Processing chunk from {filename}")
            processed_chunk = process_text_with_ollama(chunk)
            if processed_chunk:
                processed_text.append(processed_chunk)
            time.sleep(API_REQUEST_DELAY)

        if processed_text:
            final_output = "\n".join(processed_text)
            try:
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.write(final_output)
                logging.info(f"Processed file saved: {output_path}")
            except Exception as e:
                logging.error(f"Error writing to {output_path}: {e}")
        else:
            logging.warning(f"No processed text for {filename}. Not saving.")

def check_permissions(directory, permission_type):
    if permission_type == "read":
        return os.access(directory, os.R_OK)
    elif permission_type == "write":
        return os.access(directory, os.W_OK)
    return False

def ensure_output_directory(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not check_permissions(output_dir, "write"):
            logging.error(f"Permission denied for writing to {output_dir}.")
            return False
        logging.info(f"Output directory '{output_dir}' is ready.")
        return True
    except Exception as e:
        logging.error(f"Error creating or accessing output directory: {e}")
        return False

def list_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Error running nvidia-smi: {result.stderr}")
            return []
        gpus = [tuple(line.split(', ')) for line in result.stdout.strip().split('\n')]
        return gpus
    except Exception as e:
        logging.error(f"Error listing GPUs: {e}")
        return []

def select_gpu():
    gpus = list_gpus()
    if not gpus:
        logging.error("No GPUs found or unable to list GPUs.")
        return None
    print("Available GPUs:")
    for i, (index, name, utilization) in enumerate(gpus, 1):
        print(f"{i}. GPU {index}: {name} (Utilization: {utilization}%)")
    while True:
        choice = input("Select a GPU by number: ")
        if choice.strip() == "":
            logging.error("No selection made.")
            continue
        try:
            choice = int(choice)
            if 1 <= choice <= len(gpus):
                selected_gpu = gpus[choice - 1][0]
                logging.info(f"Selected GPU {selected_gpu}")
                return selected_gpu
            else:
                logging.error("Invalid selection. Try again.")
        except ValueError:
            logging.error("Invalid input. Enter a number.")

def main():
    global shutdown_requested
    selected_gpu = select_gpu()
    if not selected_gpu:
        logging.error("GPU selection failed. Exiting.")
        return
    input_dir = input("Enter source directory: ")
    if not os.path.isdir(input_dir) or not check_permissions(input_dir, "read"):
        logging.error(f"Invalid or inaccessible directory: {input_dir}. Exiting.")
        return
    output_dir = input(f"Enter output directory or press Enter for default ({DEFAULT_OUTPUT_BASE}): ").strip()
    if not output_dir:
        output_dir = os.path.join(DEFAULT_OUTPUT_BASE, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not ensure_output_directory(output_dir):
        return
    logging.info(f"Processing files in {input_dir}")
    process_files(input_dir, output_dir)
    logging.info(f"Processing complete. Outputs saved to {output_dir}")
    if shutdown_requested:
        logging.info("Graceful shutdown completed.")

if __name__ == "__main__":
    main()
