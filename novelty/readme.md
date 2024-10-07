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
