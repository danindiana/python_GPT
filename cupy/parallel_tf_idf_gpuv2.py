import os
import glob
import cupy as cp
import cupyx
from sklearn.feature_extraction.text import CountVectorizer
import argparse

# Function to load text files from a directory with error handling for encoding issues
def load_text_files(directory, max_files=1000):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))[:max_files]
    documents = []
    file_names = []
    
    for file_path in file_paths:
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

# Main function to run the whole process with multi-GPU support
def process_text_files(directory):
    # Step 1: Load text files
    documents, file_names = load_text_files(directory)
    if not documents:
        print("No text files found.")
        return

    print(f"Number of text files: {len(file_names)}")

    # Step 2: Build the term-document matrix
    tf_matrix, terms = build_term_doc_matrix(documents)

    # Split the term-document matrix by rows for two GPUs
    num_rows = tf_matrix.shape[0]
    mid_point = num_rows // 2

    tf_matrix_gpu_0 = tf_matrix[:mid_point]  # First half for GPU 0
    tf_matrix_gpu_1 = tf_matrix[mid_point:]  # Second half for GPU 1

    # Step 3: Calculate TF-IDF on GPU 0 and GPU 1 in parallel
    tf_idf_gpu_0 = calculate_tfidf_on_gpu(tf_matrix_gpu_0, 0)  # GPU 0
    tf_idf_gpu_1 = calculate_tfidf_on_gpu(tf_matrix_gpu_1, 1)  # GPU 1

    # Step 4: Combine the results from both GPUs
    combined_tf_idf = combine_tfidf_results(tf_idf_gpu_0, tf_idf_gpu_1)

    print("TF-IDF calculation complete.")

    # You can now process or display the combined TF-IDF results
    print("\nTF-IDF Matrix (non-zero values):")
    for i, term in enumerate(terms):
        if i >= combined_tf_idf.shape[0]:
            break  # Exit the loop if the index is out of range
        term_data = combined_tf_idf[i].tocoo()  # convert to COO to iterate over non-zero values
        for j, v in zip(term_data.col, term_data.data):
            if int(j) < len(file_names):
                print(f"Term: {term}, Document: {file_names[int(j)]}, TF-IDF: {v}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process text files for TF-IDF calculation using two GPUs.')
    parser.add_argument('directory', type=str, help='Directory containing the .txt files')
    
    # Parse the argument
    args = parser.parse_args()
    
    # Run the TF-IDF process
    process_text_files(args.directory)
