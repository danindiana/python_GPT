import os
import re
import pdfplumber
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def process_document(filename, directory):
    file_path = os.path.join(directory, filename)
    try:
        if filename.lower().endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        print(f"Error processing file: {filename}, Error: {str(e)}")
        return None

def sanitize_and_change_extension(filename):
    sanitized_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    return os.path.splitext(sanitized_name)[0] + ".txt"

def insert_into_database(cursor, filename, topic, top_words_str):
    try:
        cursor.execute("INSERT INTO topics (file_name, topic_id, top_words) VALUES (?, ?, ?)",
                       (sanitize_and_change_extension(filename), topic, top_words_str))
    except Exception as e:
        print(f"Error saving file {filename} to database: {str(e)}")

source_directory = "/home/jeb/Quouar_Comms"
db_path = "/home/jeb/programs/tfid_vectorizer/mydatabase.sqlite"
batch_size = 650
file_list = os.listdir(source_directory)
total_files = len(file_list)
processed_files = 0
batch_number = 0

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            file_name TEXT,
            topic_id INTEGER,
            top_words TEXT
        )
    """)
    conn.commit()

    while processed_files < total_files:
        batch_files = file_list[processed_files:min(processed_files + batch_size, total_files)]
        corpus = []
        file_names = []

        for filename in batch_files:
            text = process_document(filename, source_directory)
            if text:
                corpus.append(text)
                file_names.append(filename)

        if corpus:
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)
            nmf_model = NMF(n_components=230, random_state=1)
            nmf_dtm = nmf_model.fit_transform(tfidf_dtm)
            feature_names = tfidf_vectorizer.get_feature_names_out()

            for i, filename in enumerate(file_names):
                topic = nmf_dtm[i].argmax()
                top_words_str = ', '.join([feature_names[j] for j in nmf_model.components_[topic].argsort()[:-100 - 1:-1]])
                insert_into_database(cursor, filename, topic, top_words_str)
            conn.commit()

        processed_files += len(batch_files)
        batch_number += 1
        print(f"Completed batch {batch_number}. Total processed files: {processed_files}/{total_files}")

print("All batches processed and saved to database.")
