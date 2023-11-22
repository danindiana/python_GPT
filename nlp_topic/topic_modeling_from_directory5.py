import os
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

# Source and target directories
source_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/apache_org/"
target_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/topic_extracts/"

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Function to sanitize file names and change extension to .txt
def sanitize_and_change_extension(filename):
    sanitized_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    name_without_ext = os.path.splitext(sanitized_name)[0]  # Remove existing extension
    return f"{name_without_ext}.txt"

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    with pdfplumber.open(file_path) as pdf:
        return '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Batch processing settings
batch_size = 500  # Adjust this based on your system's capability
file_list = os.listdir(source_directory)
total_files = len(file_list)
processed_files = 0
batch_number = 0

while processed_files < total_files:
    batch_files = file_list[processed_files:min(processed_files + batch_size, total_files)]
    corpus = []
    file_names = []

    # Process each file in the batch
    for filename in batch_files:
        file_path = os.path.join(source_directory, filename)
        try:
            if filename.lower().endswith('.pdf'):
                text = read_pdf_file(file_path)
            else:
                text = read_text_file(file_path)

            if text:
                corpus.append(text)
                file_names.append(filename)
            else:
                print(f"Skipping file (no text extracted): {filename}")

        except Exception as e:
            print(f"Skipping file: {filename}, Error: {str(e)}")
            continue

    # Check if the batch is empty
    if not corpus:
        print(f"No valid files in batch {batch_number + 1}, skipping.")
        processed_files += len(batch_files)
        batch_number += 1
        continue

    # TF-IDF processing
    print(f"Processing batch {batch_number + 1}/{(total_files + batch_size - 1) // batch_size}")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)

    # NMF topic modeling
    num_topics = 100
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_dtm = nmf_model.fit_transform(tfidf_dtm)

    # Extract top words for each topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words = []

    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:-100 - 1:-1]])

    # Save results of the batch
    for i, filename in enumerate(file_names):
        topic = nmf_dtm[i].argmax()
        top_words_str = ', '.join(top_words[topic])

        result_text = f"File: {sanitize_and_change_extension(filename)}\nTopic: {topic}\nTop Words: {top_words_str}\n\n"
        result_file_path = os.path.join(target_directory, f"result_{sanitize_and_change_extension(filename)}")

        try:
            with open(result_file_path, 'w', encoding='utf-8') as result_file:
                result_file.write(result_text)
            print(f"Saved result for file: {filename}")
        except Exception as e:
            print(f"Error saving file {filename}: {str(e)}")

    processed_files += len(batch_files)
    batch_number += 1
    print(f"Completed and saved batch {batch_number}")

print(f"Total processed files: {processed_files}/{total_files}")
