import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import logging
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil  # For monitoring system memory
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Graceful Shutdown Handling
def graceful_exit(signum, frame):
    logging.info("Received termination signal. Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

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

# Function to compute cosine similarity for a chunk (to be run in parallel)
def compute_similarity_chunk(tfidf_matrix, start_idx, end_idx):
    logging.info(f"Processing chunk: {start_idx} to {end_idx}")
    
    # Compute cosine similarity for this chunk using linear_kernel
    similarity_chunk = linear_kernel(tfidf_matrix[start_idx:end_idx], tfidf_matrix)
    
    # Save the chunk to disk
    output_file = f'cosine_similarity_chunk_{start_idx}_{end_idx}.npy'
    np.save(output_file, similarity_chunk)
    logging.info(f"Saved cosine similarity chunk to {output_file}")
    
    return output_file  # Return file path for tracking

# Function to check available memory
def check_memory(user_ram_limit_gb):
    memory = psutil.virtual_memory()
    used_gb = (memory.total - memory.available) / (1024 ** 3)  # Convert to GB
    logging.info(f"Current memory usage: {used_gb:.2f} GB")
    
    # Return True if current memory usage is below the user-defined limit
    return used_gb < user_ram_limit_gb

# Function to run cosine similarity computations in parallel while checking memory usage
def compute_parallel_chunks(tfidf_matrix, chunk_size=200, max_workers=4, user_ram_limit_gb=8):
    n_samples = tfidf_matrix.shape[0]
    futures = []
    
    # Use ProcessPoolExecutor to run chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)

            # Monitor memory usage
            if not check_memory(user_ram_limit_gb):
                logging.warning(f"Memory usage exceeded the user-defined limit of {user_ram_limit_gb} GB. Pausing task submission.")
                # Wait until enough memory is available again
                while not check_memory(user_ram_limit_gb):
                    logging.info("Waiting for memory to free up...")
                    time.sleep(10)  # Wait for 10 seconds before re-checking

            # Submit each chunk as a separate job to the executor
            futures.append(executor.submit(compute_similarity_chunk, tfidf_matrix, i, end_idx))
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            result_file = future.result()
            logging.info(f"Completed processing and saved to {result_file}")

# Main program logic
def main():
    # Get user input
    input_type, path = get_input_choice()

    # Get RAM limit from the user
    while True:
        try:
            user_ram_limit_gb = float(input("Enter the maximum RAM (in GB) the program is allowed to use: ").strip())
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for RAM in GB.")

    # Step 2: Load URLs based on the choice
    if input_type == 'directory':
        url_data = load_all_urls(path)
    elif input_type == 'file':
        url_data = load_urls_from_single_file(path)
    
    # Flatten all URLs into a single list for vectorization
    logging.info("Flattening all URLs into a single list")
    all_urls = [url for urls in url_data.values() for url in urls]
    logging.info(f"Total URLs for processing: {len(all_urls)}")

    # Step 3: Vectorization for Novelty Scoring
    # Vectorize URLs using TF-IDF as a sparse matrix
    logging.info("Starting TF-IDF vectorization")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_urls)  # This is now a sparse matrix
    logging.info(f"TF-IDF vectorization completed. Shape: {tfidf_matrix.shape}")

    # Start parallel chunk-based cosine similarity computation with user-defined RAM limit
    compute_parallel_chunks(tfidf_matrix, chunk_size=300, max_workers=4, user_ram_limit_gb=user_ram_limit_gb)

    logging.info("Cosine similarity parallel chunk-based computation completed.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
