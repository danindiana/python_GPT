import os
import chardet
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
        if not self.archive:  # If the archive is empty, return a default high novelty
            return 1.0
        similarities = cosine_similarity([doc_vector], self.archive)
        return 1 - np.mean(similarities)  # Novelty is the complement of average similarity

def read_text_files(directory):
    tagged_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                with open(file_path, 'r', encoding=encoding or 'utf-8', errors='ignore') as file:
                    content = file.read()
                    tagged_data.append(TaggedDocument(words=content.split(), tags=[filename]))
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return tagged_data

def train_doc2vec_model(tagged_data):
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def score_novelty(documents, model, archive):
    for doc in documents:
        doc.vector = model.infer_vector(doc.text.split())
        doc.novelty_score = archive.calculate_novelty(doc.vector)
        archive.add_document(doc.vector)
    return documents

def main():
    text_files_directory = input("Enter the path to the directory of text files: ")
    tagged_data = read_text_files(text_files_directory)
    model = train_doc2vec_model(tagged_data)
    archive = NoveltyArchive()
    documents = [TextDocument(doc.words, doc.tags) for doc in tagged_data]
    scored_documents = score_novelty(documents, model, archive)
    scored_documents.sort(key=lambda x: x.novelty_score, reverse=True)
    
    # Output the results
    for doc in scored_documents:
        print(f"Filename: {doc.tags[0]}, Novelty Score: {doc.novelty_score}")

if __name__ == "__main__":
    main()
