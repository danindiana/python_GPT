Yup. It's another attempt to calc novelty from text files. It works but it's a kludge.


This code is a Python script designed to process large datasets of URLs, compute their TF-IDF (Term Frequency-Inverse Document Frequency) vectors, and then calculate the cosine similarity between these vectors in parallel. The script also includes mechanisms for graceful shutdown, memory monitoring, and user interaction. Below is a step-by-step explanation of the code:

### 1. **Importing Libraries**
   ```python
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
   ```
   - **os**: For interacting with the file system.
   - **pandas**: For data manipulation (though not used in this script).
   - **TfidfVectorizer**: For converting text data into TF-IDF vectors.
   - **linear_kernel**: For computing cosine similarity.
   - **numpy**: For numerical operations.
   - **logging**: For logging information and errors.
   - **signal**: For handling system signals (e.g., SIGINT, SIGTERM).
   - **sys**: For system-specific parameters and functions.
   - **concurrent.futures**: For parallel processing using `ProcessPoolExecutor` and `ThreadPoolExecutor`.
   - **psutil**: For monitoring system memory usage.
   - **time**: For time-related functions.
   - **datetime**: For time-stamping logs.

### 2. **Setting Up Logging**
   ```python
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
   ```
   - Configures the logging module to output logs at the `INFO` level with a specific format.

### 3. **Graceful Shutdown Handling**
   ```python
   graceful_shutdown = False

   def handle_shutdown_signal(signum, frame):
       global graceful_shutdown
       logging.info(f"Received shutdown signal ({signal.Signals(signum).name}). Shutting down gracefully...")
       graceful_shutdown = True

   signal.signal(signal.SIGINT, handle_shutdown_signal)
   signal.signal(signal.SIGTERM, handle_shutdown_signal)
   ```
   - **graceful_shutdown**: A global variable to manage the shutdown state.
   - **handle_shutdown_signal**: A function that sets `graceful_shutdown` to `True` when a shutdown signal (SIGINT or SIGTERM) is received.
   - **signal.signal**: Registers the `handle_shutdown_signal` function to handle SIGINT and SIGTERM signals.

### 4. **Selecting a Subdirectory**
   ```python
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
   ```
   - **select_subdirectory**: Lists subdirectories in a given directory and allows the user to select one. If no subdirectories are found, it prompts the user to enter a directory path manually.

### 5. **Reading URLs from Files**
   ```python
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
   ```
   - **read_urls_from_file**: Reads URLs from a text file and returns them as a list.

### 6. **Loading URLs from a Directory**
   ```python
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
   ```
   - **load_all_urls**: Loads URLs from all `.txt` files in a given directory and returns them as a dictionary where the keys are filenames and the values are lists of URLs.

### 7. **Loading URLs from a Single File**
   ```python
   def load_urls_from_single_file(file_path):
       logging.info(f"Loading URLs from file: {file_path}")
       urls = read_urls_from_file(file_path)
       return {os.path.basename(file_path): urls}
   ```
   - **load_urls_from_single_file**: Loads URLs from a single file and returns them as a dictionary with the filename as the key.

### 8. **Getting User Input for Directory or File**
   ```python
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
   ```
   - **get_input_choice**: Prompts the user to choose between processing a directory or a single file and returns the corresponding path.

### 9. **Thread Pool for Async Saving**
   ```python
   save_executor = ThreadPoolExecutor(max_workers=2)

   def async_save_similarity_chunk(output_file, similarity_chunk):
       np.save(output_file, similarity_chunk)
       logging.info(f"Saved cosine similarity chunk to {output_file}")
   ```
   - **save_executor**: A `ThreadPoolExecutor` for asynchronously saving computed cosine similarity chunks to avoid I/O blocking.
   - **async_save_similarity_chunk**: Saves a chunk of cosine similarity results to a file.

### 10. **Computing Cosine Similarity for a Chunk**
   ```python
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
   ```
   - **compute_similarity_chunk**: Computes the cosine similarity for a specified chunk of the TF-IDF matrix and saves the results asynchronously.

### 11. **Checking Available Memory**
   ```python
   def check_memory(user_ram_limit_gb):
       memory = psutil.virtual_memory()
       used_gb = (memory.total - memory.available) / (1024 ** 3)  # Convert to GB
       logging.info(f"Current memory usage: {used_gb:.2f} GB")
       return used_gb < user_ram_limit_gb
   ```
   - **check_memory**: Checks the current memory usage and compares it to a user-defined limit.

### 12. **Running Cosine Similarity Computations in Parallel**
   ```python
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
   ```
   - **compute_parallel_chunks**: Splits the TF-IDF matrix into chunks and computes the cosine similarity for each chunk in parallel using `ProcessPoolExecutor`. It also monitors memory usage and pauses task submission if the memory limit is exceeded.

### 13. **Main Program Logic**
   ```python
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
   ```
   - **main**: The main function that orchestrates the entire process:
     - Gets user input for whether to process a directory or a single file.
     - Loads URLs from the selected source.
     - Flattens the URLs into a single list.
     - Computes the TF-IDF vectors.
     - Computes the cosine similarity in parallel chunks.
   - **if __name__ == '__main__':** Ensures that the `main` function is called only when the script is run directly.

### Summary
This script is designed to handle large datasets of URLs by:
- Allowing the user to select a directory or a single file.
- Reading URLs from the selected source.
- Computing TF-IDF vectors for the URLs.
- Calculating cosine similarity in parallel chunks to handle large datasets efficiently.
- Monitoring memory usage and ensuring a graceful shutdown on system signals.








npz_loader_with_escape.py

This program is designed to process and analyze large datasets of URLs and their corresponding cosine similarity matrices stored in `.npz` files. The program allows the user to select specific files for processing, loads the data lazily to handle large datasets efficiently, and calculates novelty scores based on the similarity matrices. Finally, it saves the results to a CSV file. Below is a step-by-step explanation of how the program works:

### 1. **Importing Libraries**
   ```python
   import os
   import numpy as np
   import pandas as pd
   import scipy.sparse as sp
   import logging
   ```
   - **os**: For interacting with the file system.
   - **numpy**: For numerical operations.
   - **pandas**: For data manipulation and CSV writing.
   - **scipy.sparse**: For handling sparse matrices.
   - **logging**: For logging information and errors.

### 2. **Setting Up Logging**
   ```python
   logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
   ```
   - Configures the logging module to output logs at the `INFO` level with a specific format.

### 3. **Scanning for Files**
   ```python
   def scan_for_files(directory, file_extension):
       files = []
       for root, dirs, filenames in os.walk(directory):
           for filename in filenames:
               if filename.endswith(file_extension):
                   files.append(os.path.join(root, filename))
       return files
   ```
   - **scan_for_files**: Recursively scans a directory and its subdirectories for files with a specific extension (e.g., `.npz` or `.txt`).

### 4. **Parsing Flexible Ranges**
   ```python
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
   ```
   - **parse_selection**: Parses a string of flexible ranges (e.g., "1,2,4-8,11") into a list of indices.

### 5. **Selecting Files**
   ```python
   def select_files(files, file_type):
       print(f"\n{file_type} files found:")
       for idx, file in enumerate(files):
           print(f"{idx + 1}. {file}")
       
       selected_indices = input(f"Enter the numbers of the {file_type} files you want to use (e.g., 1,2,4-8,11): ").strip()
       parsed_indices = parse_selection(selected_indices)
       selected_files = [files[i - 1] for i in parsed_indices if i <= len(files)]
       return selected_files
   ```
   - **select_files**: Lists files of a specific type and allows the user to select which ones to use by entering a flexible range of indices.

### 6. **Lazy Loading of URLs**
   ```python
   def load_urls_lazily(files):
       for file_path in files:
           with open(file_path, 'r') as file:
               for line in file:
                   yield line.strip()
   ```
   - **load_urls_lazily**: Lazily loads URLs from selected text files, yielding one URL at a time to handle large datasets efficiently.

### 7. **Validating Sparse Matrices**
   ```python
   def is_valid_sparse_matrix(file_path):
       try:
           matrix = sp.load_npz(file_path)
           return sp.issparse(matrix)
       except Exception as e:
           logging.warning(f"File {file_path} is not a valid sparse matrix: {str(e)}")
           return False
   ```
   - **is_valid_sparse_matrix**: Checks if a `.npz` file contains a valid sparse matrix.

### 8. **Lazy Loading of Sparse Matrices**
   ```python
   def load_sparse_matrices_lazily(files):
       for file_path in files:
           if is_valid_sparse_matrix(file_path):
               sparse_matrix = sp.load_npz(file_path)
               logging.info(f"Loaded {file_path} with shape {sparse_matrix.shape}")
               yield sparse_matrix
           else:
               logging.warning(f"Skipping invalid sparse matrix file: {file_path}")
   ```
   - **load_sparse_matrices_lazily**: Lazily loads valid sparse matrices from selected `.npz` files, yielding one matrix at a time.

### 9. **Calculating Novelty Scores**
   ```python
   def calculate_novelty_scores_lazily(matrix_chunks):
       for chunk in matrix_chunks:
           mean_similarity = chunk.mean(axis=1)
           novelty_scores = 1 - mean_similarity.A1
           yield novelty_scores
   ```
   - **calculate_novelty_scores_lazily**: Calculates novelty scores for each chunk of similarity matrices. Novelty score is defined as `1 - mean_similarity`.

### 10. **Main Logic**
   ```python
   if __name__ == "__main__":
       base_directory = '/home/jeb/programs/rust_progs/hydra_3'

       npz_files = scan_for_files(base_directory, '.npz')
       if not npz_files:
           logging.error("No .npz files found.")
           exit()
       selected_npz_files = select_files(npz_files, "Cosine Similarity (.npz)")

       url_files = scan_for_files(base_directory, '.txt')
       if not url_files:
           logging.error("No URL files found.")
           exit()
       selected_url_files = select_files(url_files, "URL")

       url_generator = load_urls_lazily(selected_url_files)
       matrix_generator = load_sparse_matrices_lazily(selected_npz_files)

       novelty_data = []
       url_count = 0

       try:
           for matrix_chunk in matrix_generator:
               urls_in_chunk = [next(url_generator) for _ in range(matrix_chunk.shape[0])]
               novelty_scores_chunk = next(calculate_novelty_scores_lazily([matrix_chunk]))
               
               for url, score in zip(urls_in_chunk, novelty_scores_chunk):
                   novelty_data.append({'url': url, 'novelty_score': score})
                   url_count += 1
               
               logging.info(f"Processed {url_count} URLs so far.")
               
       except StopIteration:
           logging.info("All URLs and matrix chunks have been processed.")

       novelty_df = pd.DataFrame(novelty_data)
       novelty_df_sorted = novelty_df.sort_values(by='novelty_score', ascending=False)

       output_file = "sorted_novelty_scores.csv"
       chunk_size = 10000

       logging.info(f"Writing novelty scores to {output_file} in chunks of {chunk_size}...")

       try:
           with open(output_file, 'w') as f:
               for i in range(0, len(novelty_df_sorted), chunk_size):
                   novelty_df_sorted[i:i + chunk_size].to_csv(f, index=False, header=(i == 0), escapechar='\\')
                   logging.info(f"Wrote rows {i} to {i + chunk_size}")
       except Exception as e:
           logging.error(f"Error while writing CSV: {str(e)}")
       
       logging.info("Novelty scoring completed and saved to sorted_novelty_scores.csv")
   ```
   - **Main Logic**:
     1. **Scanning for Files**: Scans the base directory for `.npz` and `.txt` files.
     2. **Selecting Files**: Allows the user to select which `.npz` and `.txt` files to process.
     3. **Lazy Loading**: Lazily loads URLs and sparse matrices.
     4. **Processing Chunks**: Processes each chunk of similarity matrix and corresponding URLs to calculate novelty scores.
     5. **Saving Results**: Saves the novelty scores to a CSV file in chunks to handle large datasets efficiently.

### Summary
This program is designed to handle large datasets of URLs and their corresponding cosine similarity matrices by:
- Allowing the user to select specific files for processing.
- Loading the data lazily to handle large datasets efficiently.
- Calculating novelty scores based on the similarity matrices.
- Saving the results to a CSV file in chunks to handle large outputs efficiently.

The program is structured to be memory-efficient and scalable, making it suitable for processing large datasets.
