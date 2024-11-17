Sure, I'll refactor the code by implementing the suggested improvements. Here's the updated version of the code:

```python
import os
import glob
import cupy as cp
import cupyx
from sklearn.feature_extraction.text import CountVectorizer
import argparse

# Function to load a chunk of text files from a directory
def load_text_files(directory, start_index, chunk_size):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    file_chunk = file_paths[start_index:start_index + chunk_size]  # Get the chunk of files
    documents = []
    file_names = []

    for file_path in file_chunk:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
                file_names.append(os.path.basename(file_path))
        except UnicodeDecodeError as e:
            print(f"Error reading {file_path}: {e}")

    return documents, file_names

# Function to create a sparse term-document matrix using CountVectorizer
def build_term_doc_matrix(documents):
    vectorizer = CountVectorizer()
    term_doc_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()

    # Convert the matrix to float32 before converting to CuPy sparse matrix (CSR)
    term_doc_matrix = term_doc_matrix.astype('float32')  # Convert to float32
    term_doc_matrix_cupy = cupyx.scipy.sparse.csr_matrix(term_doc_matrix)

    return term_doc_matrix_cupy, terms

# Function to calculate TF-IDF using CuPy, assigned to a specific GPU
def calculate_tfidf_on_gpu(tf_matrix, gpu_id):
    with cp.cuda.Device(gpu_id):  # Assign computation to specific GPU
        # Ensure the matrix is on the correct GPU
        tf_matrix = tf_matrix.copy()  # Move matrix to the current device

        # Check if the matrix fits into GPU memory
        if tf_matrix.data.nbytes > cp.cuda.Device(gpu_id).mem_info[0] * 0.8:
            raise MemoryError(f"Matrix too large for GPU {gpu_id} memory.")

        # Number of documents
        num_docs = tf_matrix.shape[1]
        num_terms = tf_matrix.shape[0]  # Number of terms (rows)

        # 2. Calculate Document Frequency (DF): how many documents contain a term
        df = (tf_matrix > 0).astype(cp.float32).sum(axis=1)

        # 3. Calculate Inverse Document Frequency (IDF)
        idf = cp.log((num_docs + 1) / (df + 1)) + 1  # Smoothing to avoid division by zero

        # 4. Calculate TF-IDF by multiplying TF and IDF
        # Since IDF is a 1D array, we need to multiply row-wise in a sparse manner
        tf_idf = tf_matrix.multiply(idf.reshape(-1, 1))

        return tf_idf

# Function to combine the TF-IDF results from both GPUs (for sparse matrices)
def combine_tfidf_results(tf_idf_gpu_0, tf_idf_gpu_1):
    # Move tf_idf_gpu_1 to the same device as tf_idf_gpu_0
    with cp.cuda.Device(0):
        tf_idf_gpu_1 = tf_idf_gpu_1.copy()  # Move tf_idf_gpu_1 to GPU 0

    # Use cupyx's vstack for sparse matrices
    combined_tfidf = cupyx.scipy.sparse.vstack([tf_idf_gpu_0, tf_idf_gpu_1])
    return combined_tfidf

# Optional: Save results to a file (CSV, JSON, etc.)
def save_results_to_file(file_names, tf_idf_matrix, terms, output_file, write_header=False):
    with open(output_file, 'a', encoding='utf-8') as f:  # Use 'a' to append data
        if write_header:
            f.write("Term,Document,TF-IDF\n")
        num_terms_in_chunk = tf_idf_matrix.shape[0]  # Only iterate over terms in the current chunk
        for i in range(num_terms_in_chunk):  # Restrict iteration to terms in this chunk
            term = terms[i]
            term_data = tf_idf_matrix[i].tocoo()  # convert to COO to iterate over non-zero values
            for j, v in zip(term_data.col, term_data.data):
                if int(j) < len(file_names):
                    f.write(f"{term},{file_names[int(j)]},{v}\n")

# Main function to run the whole process with multi-GPU support
def process_text_files(directory, chunk_size, save_to_disk=False):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    if not file_paths:
        print("No .txt files found in the specified directory.")
        return

    total_files = len(file_paths)
    if chunk_size > total_files:
        chunk_size = total_files

    start_index = 0
    chunk_num = 1
    output_file = "tfidf_results.csv"
    write_header = True  # Write header for the first chunk

    while start_index < total_files:
        try:
            # Load a chunk of text files
            documents, file_names = load_text_files(directory, start_index, chunk_size)
            if not documents:
                print("No text files found in this chunk.")
                return

            print(f"Processing chunk #{chunk_num} of {len(file_names)} files, starting at index {start_index}")
            print(f"Files being processed: {file_names}")  # List file names being processed in this chunk

            # Step 2: Build the term-document matrix
            tf_matrix, terms = build_term_doc_matrix(documents)
            print(f"Term-document matrix built for chunk #{chunk_num}")

            # Split the term-document matrix by rows for two GPUs
            num_rows = tf_matrix.shape[0]
            mid_point = num_rows // 2

            tf_matrix_gpu_0 = tf_matrix[:mid_point]  # First half for GPU 0
            tf_matrix_gpu_1 = tf_matrix[mid_point:]  # Second half for GPU 1

            # Step 3: Calculate TF-IDF on GPU 0 and GPU 1 in parallel
            tf_idf_gpu_0 = calculate_tfidf_on_gpu(tf_matrix_gpu_0, 0)  # GPU 0
            print(f"TF-IDF calculated on GPU 0 for chunk #{chunk_num}")
            tf_idf_gpu_1 = calculate_tfidf_on_gpu(tf_matrix_gpu_1, 1)  # GPU 1
            print(f"TF-IDF calculated on GPU 1 for chunk #{chunk_num}")

            # Step 4: Combine the results from both GPUs
            combined_tf_idf = combine_tfidf_results(tf_idf_gpu_0, tf_idf_gpu_1)
            print(f"TF-IDF results combined for chunk #{chunk_num}")

            print(f"TF-IDF calculation for chunk #{chunk_num} is complete.")

            # Optionally save results to disk
            if save_to_disk:
                save_results_to_file(file_names, combined_tf_idf, terms, output_file, write_header)
                print(f"Results saved to {output_file}")
                write_header = False  # Do not write header for subsequent chunks

            # Move to the next chunk
            start_index += chunk_size
            chunk_num += 1
        except Exception as e:
            print(f"Error processing chunk #{chunk_num}: {e}")
            break

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process text files for TF-IDF calculation using two GPUs in chunks.')
    parser.add_argument('directory', type=str, help='Directory containing the .txt files')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Number of files to process in each chunk')
    parser.add_argument('--save_to_disk', action='store_true', help='Save results to disk (CSV)')

    # Parse the arguments
    args = parser.parse_args()

    # Run the TF-IDF process with chunking
    process_text_files(args.directory, args.chunk_size, args.save_to_disk)

    if not args.save_to_disk:
        print("Results are not being saved to disk. Use --save_to_disk to save results.")
```

### Summary of Changes:
1. **Check for No Files**: Added a check at the beginning of `process_text_files` to print a message if no `.txt` files are found.
2. **Adjust Chunk Size**: Adjusted the chunk size if it is larger than the number of files.
3. **Print Statements**: Added print statements to track the progress of the code and ensure that each step is being executed.
4. **Error Handling**: Added error handling to catch any exceptions that might be occurring during the processing of each chunk.
5. **Memory Check**: Added a memory check in `calculate_tfidf_on_gpu` to ensure the matrix fits into GPU memory.
6. **Save Results Message**: Added a message to indicate that results are not being saved if the `save_to_disk` flag is not set.

This refactored code should provide better visibility into what is happening during execution and handle potential issues more gracefully.

Sure, I'll refactor the code to include the suggested checks, print statements, and error handling. This will help identify any issues and ensure that the code produces the expected output.

Here is the refactored code:

```python
import os
import glob
import cupy as cp
import cupyx
from sklearn.feature_extraction.text import CountVectorizer
import argparse

# Function to load a chunk of text files from a directory
def load_text_files(directory, start_index, chunk_size):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    file_chunk = file_paths[start_index:start_index + chunk_size]  # Get the chunk of files
    documents = []
    file_names = []

    for file_path in file_chunk:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
                file_names.append(os.path.basename(file_path))
        except UnicodeDecodeError as e:
            print(f"Error reading {file_path}: {e}")

    return documents, file_names

# Function to create a sparse term-document matrix using CountVectorizer
def build_term_doc_matrix(documents):
    vectorizer = CountVectorizer()
    term_doc_matrix = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()

    # Convert the matrix to float32 before converting to CuPy sparse matrix (CSR)
    term_doc_matrix = term_doc_matrix.astype('float32')  # Convert to float32
    term_doc_matrix_cupy = cupyx.scipy.sparse.csr_matrix(term_doc_matrix)

    return term_doc_matrix_cupy, terms

# Function to calculate TF-IDF using CuPy, assigned to a specific GPU
def calculate_tfidf_on_gpu(tf_matrix, gpu_id):
    with cp.cuda.Device(gpu_id):  # Assign computation to specific GPU
        # Ensure the matrix is on the correct GPU
        tf_matrix = tf_matrix.copy()  # Move matrix to the current device

        # Check if the matrix fits into GPU memory
        if tf_matrix.data.nbytes > cp.cuda.Device(gpu_id).mem_info[0] * 0.8:
            raise MemoryError(f"Matrix too large for GPU {gpu_id} memory.")

        # Number of documents
        num_docs = tf_matrix.shape[1]
        num_terms = tf_matrix.shape[0]  # Number of terms (rows)

        # 2. Calculate Document Frequency (DF): how many documents contain a term
        df = (tf_matrix > 0).astype(cp.float32).sum(axis=1)

        # 3. Calculate Inverse Document Frequency (IDF)
        idf = cp.log((num_docs + 1) / (df + 1)) + 1  # Smoothing to avoid division by zero

        # 4. Calculate TF-IDF by multiplying TF and IDF
        # Since IDF is a 1D array, we need to multiply row-wise in a sparse manner
        tf_idf = tf_matrix.multiply(idf.reshape(-1, 1))

        return tf_idf

# Function to combine the TF-IDF results from both GPUs (for sparse matrices)
def combine_tfidf_results(tf_idf_gpu_0, tf_idf_gpu_1):
    # Move tf_idf_gpu_1 to the same device as tf_idf_gpu_0
    with cp.cuda.Device(0):
        tf_idf_gpu_1 = tf_idf_gpu_1.copy()  # Move tf_idf_gpu_1 to GPU 0

    # Use cupyx's vstack for sparse matrices
    combined_tfidf = cupyx.scipy.sparse.vstack([tf_idf_gpu_0, tf_idf_gpu_1])
    return combined_tfidf

# Optional: Save results to a file (CSV, JSON, etc.)
def save_results_to_file(file_names, tf_idf_matrix, terms, output_file, write_header=False):
    with open(output_file, 'a', encoding='utf-8') as f:  # Use 'a' to append data
        if write_header:
            f.write("Term,Document,TF-IDF\n")
        num_terms_in_chunk = tf_idf_matrix.shape[0]  # Only iterate over terms in the current chunk
        for i in range(num_terms_in_chunk):  # Restrict iteration to terms in this chunk
            term = terms[i]
            term_data = tf_idf_matrix[i].tocoo()  # convert to COO to iterate over non-zero values
            for j, v in zip(term_data.col, term_data.data):
                if int(j) < len(file_names):
                    f.write(f"{term},{file_names[int(j)]},{v}\n")

# Main function to run the whole process with multi-GPU support
def process_text_files(directory, chunk_size, save_to_disk=False):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    if not file_paths:
        print("No .txt files found in the specified directory.")
        return

    total_files = len(file_paths)
    if chunk_size > total_files:
        chunk_size = total_files

    start_index = 0
    chunk_num = 1
    output_file = "tfidf_results.csv"
    write_header = True  # Write header for the first chunk

    while start_index < total_files:
        try:
            # Load a chunk of text files
            documents, file_names = load_text_files(directory, start_index, chunk_size)
            if not documents:
                print("No text files found in this chunk.")
                return

            print(f"Processing chunk #{chunk_num} of {len(file_names)} files, starting at index {start_index}")
            print(f"Files being processed: {file_names}")  # List file names being processed in this chunk

            # Step 2: Build the term-document matrix
            tf_matrix, terms = build_term_doc_matrix(documents)
            print(f"Term-document matrix built for chunk #{chunk_num}")

            # Split the term-document matrix by rows for two GPUs
            num_rows = tf_matrix.shape[0]
            mid_point = num_rows // 2

            tf_matrix_gpu_0 = tf_matrix[:mid_point]  # First half for GPU 0
            tf_matrix_gpu_1 = tf_matrix[mid_point:]  # Second half for GPU 1

            # Step 3: Calculate TF-IDF on GPU 0 and GPU 1 in parallel
            tf_idf_gpu_0 = calculate_tfidf_on_gpu(tf_matrix_gpu_0, 0)  # GPU 0
            print(f"TF-IDF calculated on GPU 0 for chunk #{chunk_num}")
            tf_idf_gpu_1 = calculate_tfidf_on_gpu(tf_matrix_gpu_1, 1)  # GPU 1
            print(f"TF-IDF calculated on GPU 1 for chunk #{chunk_num}")

            # Step 4: Combine the results from both GPUs
            combined_tf_idf = combine_tfidf_results(tf_idf_gpu_0, tf_idf_gpu_1)
            print(f"TF-IDF results combined for chunk #{chunk_num}")

            print(f"TF-IDF calculation for chunk #{chunk_num} is complete.")

            # Optionally save results to disk
            if save_to_disk:
                save_results_to_file(file_names, combined_tf_idf, terms, output_file, write_header)
                print(f"Results saved to {output_file}")
                write_header = False  # Do not write header for subsequent chunks

            # Move to the next chunk
            start_index += chunk_size
            chunk_num += 1
        except Exception as e:
            print(f"Error processing chunk #{chunk_num}: {e}")
            break

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process text files for TF-IDF calculation using two GPUs in chunks.')
    parser.add_argument('directory', type=str, help='Directory containing the .txt files')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Number of files to process in each chunk')
    parser.add_argument('--save_to_disk', action='store_true', help='Save results to disk (CSV)')

    # Parse the arguments
    args = parser.parse_args()

    # Run the TF-IDF process with chunking
    process_text_files(args.directory, args.chunk_size, args.save_to_disk)

    if not args.save_to_disk:
        print("Results are not being saved to disk. Use --save_to_disk to save results.")
```
With these changes, the code should provide more insight into its execution and help identify any issues that might be preventing it from producing the expected output.

So we have a large corpus of text files from web scraping. We would like to organize them based on relevance, topicality, and overall information quality for enrollment in a RAG (retrieval augmented generator using LLM) vector database (as embeddings). Using cupy and NLP what are some ways we can sort these many text files so that we can exclude undesirable (noise) from desirable (signal)?
