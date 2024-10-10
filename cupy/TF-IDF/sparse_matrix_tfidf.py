import os
import glob
import cupy as cp
import cupyx
from sklearn.feature_extraction.text import CountVectorizer
import argparse

# Function to load text files from a directory
def load_text_files(directory):
    file_paths = glob.glob(os.path.join(directory, "*.txt"))
    documents = []
    file_names = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
            file_names.append(os.path.basename(file_path))
    
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

# Function to calculate TF-IDF using CuPy
def calculate_tfidf(tf_matrix):
    # Number of documents
    num_docs = tf_matrix.shape[1]
    num_terms = tf_matrix.shape[0]  # Number of terms (rows)
    
    print(f"TF Matrix Shape: {tf_matrix.shape}")  # Debugging output

    # 2. Calculate Document Frequency (DF): how many documents contain a term
    df = (tf_matrix > 0).astype(cp.float32).sum(axis=1)

    # 3. Calculate Inverse Document Frequency (IDF)
    idf = cp.log((num_docs + 1) / (df + 1)) + 1  # Smoothing to avoid division by zero

    # 4. Calculate TF-IDF by multiplying TF and IDF
    # Since IDF is a 1D array, we need to multiply row-wise in a sparse manner
    tf_idf = tf_matrix.multiply(idf.reshape(-1, 1))

    return tf_idf, num_terms

# Main function to run the whole process
def process_text_files(directory):
    # Step 1: Load text files
    documents, file_names = load_text_files(directory)
    if not documents:
        print("No text files found.")
        return

    print(f"Number of text files: {len(file_names)}")  # Debugging output

    # Step 2: Build the term-document matrix
    tf_matrix, terms = build_term_doc_matrix(documents)

    # Step 3: Calculate TF-IDF
    tf_idf_matrix, num_terms = calculate_tfidf(tf_matrix)

    # Debugging: Print the number of terms and ensure it aligns with terms list
    print(f"Number of terms: {len(terms)}, TF-IDF matrix rows: {num_terms}")

    # Ensure the iteration only goes over the valid range of terms
    num_terms = min(len(terms), num_terms)

    # Print the results (non-zero entries)
    print("\nTF-IDF Matrix (non-zero values):")
    for i in range(num_terms):
        term = terms[i]
        term_data = tf_idf_matrix[i].tocoo()  # convert to COO to iterate over non-zero values
        for j, v in zip(term_data.col, term_data.data):
            # Only print valid indices for documents within the range of file_names
            if int(j) < len(file_names):
                print(f"Term: {term}, Document: {file_names[int(j)]}, TF-IDF: {v}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process text files for TF-IDF calculation.')
    parser.add_argument('directory', type=str, help='Directory containing the .txt files')
    
    # Parse the argument
    args = parser.parse_args()
    
    # Run the TF-IDF process
    process_text_files(args.directory)
