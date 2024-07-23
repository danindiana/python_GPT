import os
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import signal
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ask the user for source and target directories
source_directory = input("Enter the source directory (where your files are): ")
target_directory = input("Enter the target directory (where you want to save the results): ")

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

def sanitize_and_change_extension(filename):
    """Cleans and renames files, changing the extension to .txt."""
    sanitized_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    name_without_ext = os.path.splitext(sanitized_name)[0]
    return f"{name_without_ext}.txt"

def read_text_file(file_path):
    """Reads text from a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    """Extracts text from a PDF (without image extraction)."""
    with pdfplumber.open(file_path) as pdf:
        return '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])

def signal_handler(sig, frame):
    """Handles graceful shutdown on receiving a signal."""
    logging.info("Gracefully shutting down...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

batch_size = 500  # Adjust based on your system's capability
file_list = os.listdir(source_directory)
total_files = len(file_list)
processed_files = 0
batch_number = 0

while processed_files < total_files:
    batch_files = file_list[processed_files:min(processed_files + batch_size, total_files)]
    corpus = []
    file_names = []

    logging.info(f"Starting batch {batch_number + 1} processing...")

    for filename in batch_files:
        file_path = os.path.join(source_directory, filename)
        logging.info(f"Reading file: {filename}")

        try:
            if filename.lower().endswith('.pdf'):
                text = read_pdf_file(file_path)
                logging.info(f"Processed PDF file: {filename}")
            else:
                text = read_text_file(file_path)
                logging.info(f"Processed text file: {filename}")

            if text:
                corpus.append(text)
                file_names.append(filename)
            else:
                logging.warning(f"Skipping file (no text extracted): {filename}")

        except Exception as e:
            logging.error(f"Skipping file: {filename}, Error: {str(e)}")
            continue

    if not corpus:
        logging.warning(f"No valid files in batch {batch_number + 1}, skipping.")
        processed_files += len(batch_files)
        batch_number += 1
        continue

    logging.info(f"Performing TF-IDF processing for batch {batch_number + 1}...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)

    logging.info(f"Performing NMF topic modeling for batch {batch_number + 1}...")
    num_topics = 100
    nmf_model = NMF(n_components=num_topics, random_state=1, alpha_W=0.1, alpha_H=0.1)  # Use alpha_W and alpha_H for regularization

    # Fit the NMF model on the entire batch
    nmf_model.fit(tfidf_dtm)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:-100 - 1:-1]])

    logging.info(f"Saving results for batch {batch_number + 1}...")
    nmf_dtm = nmf_model.transform(tfidf_dtm)
    for i, filename in enumerate(file_names):
        topic = nmf_dtm[i].argmax()
        top_words_str = ', '.join(top_words[topic])

        result_text = f"File: {sanitize_and_change_extension(filename)}\nTopic: {topic}\nTop Words: {top_words_str}\n\n"
        result_file_path = os.path.join(target_directory, f"result_{sanitize_and_change_extension(filename)}")

        try:
            with open(result_file_path, 'w', encoding='utf-8') as result_file:
                result_file.write(result_text)
            logging.info(f"Saved result for file: {filename}")
        except Exception as e:
            logging.error(f"Error saving file {filename}: {str(e)}")

    processed_files += len(batch_files)
    batch_number += 1
    logging.info(f"Completed batch {batch_number}. Total processed files: {processed_files}/{total_files}")

logging.info("All batches processed.")
