from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import cosine
from collections import deque
import os
import numpy as np

# Define a class to represent individual text documents
class TextDocument:
    def __init__(self, text, tags):
        self.text = text
        self.tags = tags
        self.vector = None
        self.novelty_score = 0

# Define a class for the novelty archive
class NoveltyArchive:
    def __init__(self, max_size):
        self.archive = deque(maxlen=max_size)

    def add_document(self, document):
        self.archive.append(document.vector)

    def calculate_novelty(self, document, k=15):
        if not self.archive:
            return 0
        distances = [cosine(document.vector, other_vector) for other_vector in self.archive]
        nearest_neighbors = sorted(distances)[:k]
        novelty_score = sum(nearest_neighbors) / k
        return novelty_score

# Function to train a Doc2Vec model on the text documents
def train_doc2vec_model(tagged_data):
    model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Function to read text files and convert them into TaggedDocuments
def read_text_files(directory):
    tagged_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                tagged_data.append(TaggedDocument(words=content.split(), tags=[filename]))
    return tagged_data

# Directory containing the text files
text_files_directory = 'path/to/text/files'

# Read and prepare the text files
tagged_data = read_text_files(text_files_directory)

# Train the Doc2Vec model
doc2vec_model = train_doc2vec_model(tagged_data)

# Create the text documents with their vectors
documents = [TextDocument(text=td.words, tags=td.tags) for td in tagged_data]
for document in documents:
    document.vector = doc2vec_model.infer_vector(document.text)

# Create the novelty archive
archive = NoveltyArchive(max_size=100)

# Calculate novelty scores for each document
for document in documents:
    document.novelty_score = archive.calculate_novelty(document)
    archive.add_document(document)  # Optionally add the document to the archive

# Sort the documents by their novelty score
documents.sort(key=lambda x: x.novelty_score, reverse=True)

# Output the novelty scores
for document in documents:
    print(f"Document: {document.tags[0]}, Novelty Score: {document.novelty_score}")
