import os
import chardet
import threading
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from datetime import datetime

class TextDocument:
    def __init__(self, text, tags):
        self.text = text
        self.tags = tags
        self.vector = None
        self.novelty_score = None

class NoveltyArchive:
    def __init__(self):
        self.archive = []

    def add_document(self, doc_vector):
        self.archive.append(doc_vector)

    def calculate_novelty(self, doc_vector):
        if not self.archive:
            return 1.0
        similarities = cosine_similarity([doc_vector], self.archive)
        return 1 - np.mean(similarities)

def read_text_files(directory):
    tagged_data = []
    file_count = 0
    error_count = 0
    total_files = len([name for name in os.listdir(directory) if name.endswith('.txt')])
    print(f"Found {total_files} text files. Starting to process...")

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                with open(file_path, 'r', encoding=encoding or 'utf-8', errors='ignore') as file:
                    content = file.read()
                    tagged_data.append(TaggedDocument(words=content.split(), tags=[filename]))
                    file_count += 1
                    print(f"Successfully processed file {file_count} of {total_files}: {filename}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                error_count += 1
    print(f"Total files processed: {file_count}, Errors: {error_count}")
    return tagged_data

def train_doc2vec_model(tagged_data):
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    print("Starting model training...")
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    print("Model training completed.")
    return model

def score_novelty(documents, model, archive):
    print("Scoring novelty for each document...")
    for index, doc in enumerate(documents):
        if isinstance(doc.text, list):
            doc.text = ' '.join(doc.text)
        doc.vector = model.infer_vector(doc.text.split())
        doc.novelty_score = archive.calculate_novelty(doc.vector)
        archive.add_document(doc.vector)
        print(f"Processed {index + 1}/{len(documents)}: {doc.tags[0]} with novelty score: {doc.novelty_score:.4f}")
    print("All documents have been processed and scored.")
    return documents

def write_results_to_file(scored_documents, interval=300):
    while True:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"novelty_scores_{timestamp}.txt"
        with open(filename, 'w') as f:
            for doc in scored_documents:
                f.write(f"Filename: {doc.tags[0]}, Novelty Score: {doc.novelty_score:.4f}\n")
        print(f"Results written to {filename}")
        time.sleep(interval)

def main():
    start_time = time.time()
    
    text_files_directory = input("Enter the path to the directory of text files: ").strip()
    
    if not text_files_directory or not os.path.isdir(text_files_directory):
        print("The specified directory does not exist or was not entered. Exiting the program.")
        return
    
    print("Reading text files...")
    read_start_time = time.time()
    tagged_data = read_text_files(text_files_directory)
    read_end_time = time.time()
    print(f"Finished reading files in {read_end_time - read_start_time:.2f} seconds.")
    
    if not tagged_data:
        print("No text files found or an error occurred. Exiting the program.")
        return
    
    print("Training Doc2Vec model...")
    training_start_time = time.time()
    model = train_doc2vec_model(tagged_data)
    training_end_time = time.time()
    print(f"Model trained in {training_end_time - training_start_time:.2f} seconds.")
    
    archive = NoveltyArchive()
    documents = [TextDocument(' '.join(doc.words), doc.tags) for doc in tagged_data]
    
    print("Scoring documents for novelty...")
    scoring_start_time = time.time()
    scored_documents = score_novelty(documents, model, archive)
    scoring_end_time = time.time()
    print(f"Documents scored in {scoring_end_time - scoring_start_time:.2f} seconds.")
    
    scored_documents.sort(key=lambda x: x.novelty_score, reverse=True)
    
    print("Novelty scores:")
    for doc in scored_documents:
        print(f"Filename: {doc.tags[0]}, Novelty Score: {doc.novelty_score:.4f}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    # Start the timer to write results to file at regular intervals
    write_thread = threading.Thread(target=write_results_to_file, args=(scored_documents,))
    write_thread.daemon = True
    write_thread.start()

    # Suggest a file name upon completion
    completion_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    suggested_filename = f"final_novelty_scores_{completion_timestamp}.txt"
    print(f"Suggested filename for final output: {suggested_filename}")

if __name__ == "__main__":
    main()
