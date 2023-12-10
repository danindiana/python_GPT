import os
import re
import pdfplumber
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Source directory
source_directory = "/your/source_directory/"

# Create and connect to the SQLite database
db_path = "/your/data-base-path/mydatabase.db"  # Update with the desired path to your database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a table for the results
cursor.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        file_name TEXT,
        topic_id INTEGER,
        top_words TEXT
    )
""")
conn.commit()

def sanitize_and_change_extension(filename):
    sanitized_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    name_without_ext = os.path.splitext(sanitized_name)[0]
    return f"{name_without_ext}.txt"

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    with pdfplumber.open(file_path) as pdf:
        return '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])

batch_size = 350  # Adjust based on your system's capability
file_list = os.listdir(source_directory)
total_files = len(file_list)
processed_files = 0
batch_number = 0

while processed_files < total_files:
    batch_files = file_list[processed_files:min(processed_files + batch_size, total_files)]
    corpus = []
    file_names = []

    print(f"Starting batch {batch_number + 1} processing...")

    for filename in batch_files:
        file_path = os.path.join(source_directory, filename)
        print(f"Reading file: {filename}")

        try:
            if filename.lower().endswith('.pdf'):
                text = read_pdf_file(file_path)
                print(f"Processed PDF file: {filename}")
            else:
                text = read_text_file(file_path)
                print(f"Processed text file: {filename}")

            if text:
                corpus.append(text)
                file_names.append(filename)
            else:
                print(f"Skipping file (no text extracted): {filename}")

        except Exception as e:
            print(f"Skipping file: {filename}, Error: {str(e)}")
            continue

    if not corpus:
        print(f"No valid files in batch {batch_number + 1}, skipping.")
        processed_files += len(batch_files)
        batch_number += 1
        continue

    print(f"Performing TF-IDF processing for batch {batch_number + 1}...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)

    print("Performing NMF topic modeling...")
    num_topics = 200
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_dtm = nmf_model.fit_transform(tfidf_dtm)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words = []

    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:-100 - 1:-1]])

    print(f"Saving results for batch {batch_number + 1}...")
    for i, filename in enumerate(file_names):
        topic = nmf_dtm[i].argmax()
        top_words_str = ', '.join(top_words[topic])

        # Insert results into the database
        try:
            cursor.execute("INSERT INTO topics (file_name, topic_id, top_words) VALUES (?, ?, ?)",
                           (sanitize_and_change_extension(filename), topic, top_words_str))
            conn.commit()
            print(f"Saved result for file: {filename}")
        except Exception as e:
            print(f"Error saving file {filename} to database: {str(e)}")

    processed_files += len(batch_files)
    batch_number += 1
    print(f"Completed batch {batch_number}. Total processed files: {processed_files}/{total_files}")

conn.close()
print("All batches processed and saved to the database.")
