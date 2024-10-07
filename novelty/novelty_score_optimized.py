import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import logging
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
import psutil  # For monitoring system memory
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to manage graceful shutdown
graceful_shutdown = False

# Graceful Shutdown Handling
def handle_shutdown_signal(signum, frame):
    global graceful_shutdown
    logging.info(f"Received shutdown signal ({signal.Signals(signum).name}). Shutting down gracefully...")
    graceful_shutdown = True

# Register shutdown signals (SIGINT and SIGTERM)
signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

# Function to list subdirectories and allow user to select
def select_subdirectory(default_directory):
    subdirs = [d for d in os.listdir(default_directory) if os.path.isdir(os.path.join(default_directory, d))]
    
    if not subdirs:
        logging.info(f"No subdirectories found in {default_directory}. Please enter a directory path manually.")
        return input("Enter the full directory path: ").strip()
    
    print(f"Subdirectories in {default_directory}:")
    for idx, subdir in enumerate(subdirs):
        print(f"{idx + 1}. {subdir}")
    
    while True:
        try:
            choice = int(input(f"Select a subdirectory (1-{len(subdirs)}): ").strip())
            if 1 <= choice <= len(subdirs):
                selected_dir = os.path.join(default_directory, subdirs[choice - 1])
                logging.info(f"Selected directory: {selected_dir}")
                return selected_dir
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(subdirs)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(subdirs)}.")

# Function to read URLs from large text files
def read_urls_from_file(file_path):
    urls = []
    logging.info(f"Reading URLs from {file_path}")
    try:
        with open(file_path, 'r') as file:
            for line in file:
                url = line.strip()
                if url:
                    urls.append(url)
        logging.info(f"Finished reading {len(urls)} URLs from {file_path}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
    return urls

# Load URLs from directory
def load_all_urls(directory):
    logging.info(f"Checking directory path: {directory}")
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        raise FileNotFoundError(f"No such file or directory: {directory}")
    
    url_data = {}
    logging.info(f"Loading URLs from files in directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            urls = read_urls_from_file(file_path)
            url_data[filename] = urls
            logging.info(f"{filename}: Loaded {len(urls)} URLs")
    logging.info(f"Loaded all URLs from directory. Total files processed: {len(url_data)}")
    return url_data

# Load URLs from a single file
def load_urls_from_single_file(file_path):
    logging.info(f"Loading URLs from file: {file_path}")
    urls = read_urls_from_file(file_path)
    return {os.path.basename(file_path): urls}

# Function to get user input for directory or file
def get_input_choice(default_directory='/home/jeb/programs/rust_progs/hydra_3'):
    choice = input("Do you want to process a (D)irectory or a (S)ingle file? [D/S]: ").strip().lower()
    if choice == 'd':
        selected_directory = select_subdirectory(default_directory)
        return 'directory', selected_directory
    elif choice == 's':
        return 'file', input("Enter the file path: ").strip()
    else:
        logging.error("Invalid choice. Please select either 'D' for directory or 'S' for single file.")
        sys.exit(1)

# Thread pool for async saving
save_executor = ThreadPoolExecutor(max_workers=2)

def async_save_similarity_chunk(output_file, similarity_chunk):
    np.save(output_file, similarity_chunk)
    logging.info(f"Saved cosine similarity chunk to {output_file}")

# Function to compute cosine similarity for a chunk (to be run in parallel)
def compute_similarity_chunk(tfidf_matrix, start_idx, end_idx, chunk_num, total_chunks):
    if graceful_shutdown:
        logging.info("Graceful shutdown initiated. Exiting the current task.")
        return

    start_time = datetime.now()
    logging.info(f"Processing chunk: {start_idx} to {end_idx} (Chunk {chunk_num} of {total_chunks})")

    # Actual computation
    similarity_chunk = linear_kernel(tfidf_matrix[start_idx:end_idx], tfidf_matrix)
    
    # Log the progress after computation
    processing_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Finished processing chunk {chunk_num} | Time Elapsed: {processing_time:.2f} seconds")
    
    # Save asynchronously to avoid I/O blocking
    output_file = f'cosine_similarity_chunk_{start_idx}_{end_idx}.npy'
    save_executor.submit(async_save_similarity_chunk, output_file, similarity_chunk)
    
    return output_file

# Function to check available memory
def check_memory(user_ram_limit_gb):
    memory = psutil.virtual_memory()
    used_gb = (memory.total - memory.available) / (1024 ** 3)  # Convert to GB
    logging.info(f"Current memory usage: {used_gb:.2f} GB")
    return used_gb < user_ram_limit_gb

# Function to run cosine similarity computations in parallel while checking memory usage
def compute_parallel_chunks(tfidf_matrix, chunk_size=2000, max_workers=8, user_ram_limit_gb=8):
    n_samples = tfidf_matrix.shape[0]
    total_chunks = n_samples // chunk_size + (n_samples % chunk_size > 0)  # Total number of chunks
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            for i in range(0, n_samples, chunk_size):
                if graceful_shutdown:
                    logging.info("Graceful shutdown detected. Terminating task submissions.")
                    break

                end_idx = min(i + chunk_size, n_samples)
                chunk_num = (i // chunk_size) + 1  # Chunk number
                
                if not check_memory(user_ram_limit_gb):
                    logging.warning(f"Memory usage exceeded the user-defined limit of {user_ram_limit_gb} GB. Pausing task submission.")
                    while not check_memory(user_ram_limit_gb):
                        logging.info("Waiting for memory to free up...")
                        time.sleep(10)

                futures.append(executor.submit(compute_similarity_chunk, tfidf_matrix, i, end_idx, chunk_num, total_chunks))

            # Wait for all futures to complete or cancel them if shutdown was initiated
            for future in as_completed(futures):
                if graceful_shutdown:
                    logging.info("Cancelling all pending tasks due to shutdown...")
                    for future in futures:
                        future.cancel()
                    break
                result_file = future.result()
                logging.info(f"Completed processing and saved to {result_file}")
        
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")

# Main program logic
def main():
    input_type, path = get_input_choice()

    while True:
        try:
            user_ram_limit_gb = float(input("Enter the maximum RAM (in GB) the program is allowed to use: ").strip())
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for RAM in GB.")

    if input_type == 'directory':
        url_data = load_all_urls(path)
    elif input_type == 'file':
        url_data = load_urls_from_single_file(path)

    logging.info("Flattening all URLs into a single list")
    all_urls = [url for urls in url_data.values() for url in urls]
    logging.info(f"Total URLs for processing: {len(all_urls)}")

    logging.info("Starting TF-IDF vectorization")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_urls)  
    logging.info(f"TF-IDF vectorization completed. Shape: {tfidf_matrix.shape}")

    compute_parallel_chunks(tfidf_matrix, chunk_size=2000, max_workers=8, user_ram_limit_gb=user_ram_limit_gb)
    logging.info("Cosine similarity parallel chunk-based computation completed.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
