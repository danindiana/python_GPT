import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to scan and list files in a directory and its subdirectories
def scan_for_files(directory, file_extension):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

# Function to parse flexible ranges for selection (e.g., "1,2,4-8,11")
def parse_selection(selection):
    indices = []
    parts = selection.split(',')
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices

# Function to ask user to select files with flexible input
def select_files(files, file_type):
    print(f"\n{file_type} files found:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    
    selected_indices = input(f"Enter the numbers of the {file_type} files you want to use (e.g., 1,2,4-8,11): ").strip()
    parsed_indices = parse_selection(selected_indices)
    selected_files = [files[i - 1] for i in parsed_indices if i <= len(files)]
    return selected_files

# Lazy loading of URLs from selected files
def load_urls_lazily(files):
    for file_path in files:
        with open(file_path, 'r') as file:
            for line in file:
                yield line.strip()

# Validate if an npz file contains a sparse matrix
def is_valid_sparse_matrix(file_path):
    try:
        matrix = sp.load_npz(file_path)
        return sp.issparse(matrix)
    except Exception as e:
        logging.warning(f"File {file_path} is not a valid sparse matrix: {str(e)}")
        return False

# Load and combine sparse similarity matrices lazily
def load_sparse_matrices_lazily(files):
    for file_path in files:
        if is_valid_sparse_matrix(file_path):
            sparse_matrix = sp.load_npz(file_path)
            logging.info(f"Loaded {file_path} with shape {sparse_matrix.shape}")
            yield sparse_matrix
        else:
            logging.warning(f"Skipping invalid sparse matrix file: {file_path}")

# Calculate novelty scores lazily from similarity matrix chunks
def calculate_novelty_scores_lazily(matrix_chunks):
    for chunk in matrix_chunks:
        # Mean similarity for each URL in this chunk (axis=1 computes the row-wise mean)
        mean_similarity = chunk.mean(axis=1)
        # Novelty score is 1 - mean similarity
        novelty_scores = 1 - mean_similarity.A1  # .A1 converts sparse matrix to 1D array
        yield novelty_scores

# Main logic
if __name__ == "__main__":
    base_directory = '/home/jeb/programs/rust_progs/hydra_3'

    # Scan for .npz files and ask the user to select which ones to use
    npz_files = scan_for_files(base_directory, '.npz')
    if not npz_files:
        logging.error("No .npz files found.")
        exit()
    selected_npz_files = select_files(npz_files, "Cosine Similarity (.npz)")

    # Scan for URL files and ask the user to select which ones to use
    url_files = scan_for_files(base_directory, '.txt')
    if not url_files:
        logging.error("No URL files found.")
        exit()
    selected_url_files = select_files(url_files, "URL")

    # Lazy load URLs and matrices
    url_generator = load_urls_lazily(selected_url_files)
    matrix_generator = load_sparse_matrices_lazily(selected_npz_files)

    # Open a CSV writer for saving results
    novelty_data = []
    url_count = 0

    # Process chunks of similarity matrix and URLs lazily
    try:
        for matrix_chunk in matrix_generator:
            urls_in_chunk = [next(url_generator) for _ in range(matrix_chunk.shape[0])]
            novelty_scores_chunk = next(calculate_novelty_scores_lazily([matrix_chunk]))
            
            # Store the URL and corresponding novelty score
            for url, score in zip(urls_in_chunk, novelty_scores_chunk):
                novelty_data.append({'url': url, 'novelty_score': score})
                url_count += 1
            
            logging.info(f"Processed {url_count} URLs so far.")
            
    except StopIteration:
        logging.info("All URLs and matrix chunks have been processed.")

    # Convert to DataFrame and sort by novelty score
    novelty_df = pd.DataFrame(novelty_data)
    novelty_df_sorted = novelty_df.sort_values(by='novelty_score', ascending=False)

    # Save the sorted novelty scores to a CSV file, handling special characters with escapechar
    output_file = "sorted_novelty_scores.csv"
    chunk_size = 10000  # Number of rows per chunk

    logging.info(f"Writing novelty scores to {output_file} in chunks of {chunk_size}...")

    try:
        # Open the file once, write in chunks with an escape character for special symbols
        with open(output_file, 'w') as f:
            for i in range(0, len(novelty_df_sorted), chunk_size):
                novelty_df_sorted[i:i + chunk_size].to_csv(f, index=False, header=(i == 0), escapechar='\\')
                logging.info(f"Wrote rows {i} to {i + chunk_size}")
    except Exception as e:
        logging.error(f"Error while writing CSV: {str(e)}")
    
    logging.info("Novelty scoring completed and saved to sorted_novelty_scores.csv")
