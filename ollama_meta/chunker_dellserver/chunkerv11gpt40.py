import os
import requests
import datetime
import json
import logging

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # Corrected endpoint
MODEL_NAME = "qwen2.5-coder:3b"  # Updated model name
CHUNK_SIZE = 1024  # Characters per chunk (adjust as needed)
DEFAULT_OUTPUT_BASE = "/home/processed_output"  # Updated base directory for default output

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_text_with_ollama(prompt):
    """Send a prompt to the Ollama API and return the response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,  # Adjust temperature as needed
        "max_tokens": 100  # Adjust max tokens as needed
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
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

def chunk_text(text, chunk_size):
    """Split text into chunks of a specified size."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

def process_files(input_dir, output_dir):
    """Iterate over text files in input_dir, process them, and save outputs."""
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):  # Process only .txt files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as infile:
                text = infile.read()

            # Process the file in chunks if needed
            processed_text = []
            for chunk in chunk_text(text, CHUNK_SIZE):
                logging.info(f"Processing chunk from {filename}...")
                processed_chunk = process_text_with_ollama(chunk)
                logging.debug(f"Processed chunk: {processed_chunk}")  # Print the processed text
                if processed_chunk:  # Only append non-empty responses
                    processed_text.append(processed_chunk)

            # Save the processed output
            if processed_text:  # Only save if there's processed text
                final_output = "\n".join(processed_text)
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.write(final_output)
                logging.info(f"Processed file saved: {output_path}")
                print(f"Final content written to {output_path}:\n{final_output}")
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

def main():
    # Step 1: Prompt user for source directory
    input_dir = input("Enter the source directory containing text files: ").strip()
    if not os.path.isdir(input_dir):
        logging.error(f"Error: {input_dir} is not a valid directory. Exiting.")
        return

    if not check_permissions(input_dir, "read"):
        logging.error(f"Error: Permission denied for reading files in {input_dir}. Exiting.")
        return

    # Step 2: Prompt for or create output directory
    default_output_dir = os.path.join(DEFAULT_OUTPUT_BASE, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output_dir = input(f"Enter output directory or press Enter to use default ({default_output_dir}): ").strip()
    if not output_dir:
        output_dir = default_output_dir

    if not ensure_output_directory(output_dir):
        return

    # Step 3: Process files
    logging.info(f"\nProcessing files in {input_dir}...")
    process_files(input_dir, output_dir)
    logging.info(f"\nProcessing complete! Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
