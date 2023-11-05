import os
import chardet
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    for index, doc in enumerate(documents):
        # Ensure that doc.text is a string before splitting
        if isinstance(doc.text, list):
            doc.text = ' '.join(doc.text)
        doc.vector = model.infer_vector(doc.text.split())
        doc.novelty_score = archive.calculate_novelty(doc.vector)
        archive.add_document(doc.vector)
        print(f"Processed {index + 1}/{len(documents)}: {doc.tags[0]} with novelty score: {doc.novelty_score}")
    return documents

def main():
    text_files_directory = input("Enter the path to the directory of text files: ").strip()
    
    # Check if the input is not empty
    if not text_files_directory:
        print("No directory was entered. Exiting the program.")
        return
    
    print("Reading text files...")
    tagged_data = read_text_files(text_files_directory)
    
    if not tagged_data:
        print("No text files found or an error occurred. Exiting the program.")
        return
    
    print("Training Doc2Vec model...")
    model = train_doc2vec_model(tagged_data)
    archive = NoveltyArchive()
    
    # Ensure that the text for each TextDocument is a single string
    documents = [TextDocument(' '.join(doc.words), doc.tags) for doc in tagged_data]
    
    print("Scoring documents for novelty...")
    scored_documents = score_novelty(documents, model, archive)
    scored_documents.sort(key=lambda x: x.novelty_score, reverse=True)
    
    # Output the results
    print("Novelty scores:")
    for doc in scored_documents:
        print(f"Filename: {doc.tags[0]}, Novelty Score: {doc.novelty_score}")

if __name__ == "__main__":
    main()
