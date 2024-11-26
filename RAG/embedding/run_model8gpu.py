import os
import json
from sentence_transformers import SentenceTransformer
import torch

# Step 1: Choose a pre-trained model
model_name = "all-MiniLM-L6-v2"

# Step 2: Load the pre-trained SentenceTransformer model
# Utilize GPUs by setting device="cuda"
model = SentenceTransformer(model_name, device="cuda")

# Step 3: Check the number of GPUs available
print(f"Number of GPUs available: {torch.cuda.device_count()}")

# Step 4: Prompt the user for the directory
directory = input("Enter the directory containing text or JSON files: ")

# Step 5: Recursively process files in the directory
def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".txt"):
                process_text_file(file_path)
            elif file.endswith(".json"):
                process_json_file(file_path)

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()
        sentences = text_content.split('\n')
        generate_embeddings(sentences)

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_content = json.load(file)
        # Assuming the JSON structure is a list of sentences or a dictionary with a 'text' key
        if isinstance(json_content, list):
            sentences = json_content
        elif isinstance(json_content, dict) and 'text' in json_content:
            sentences = json_content['text'].split('\n')
        else:
            print(f"Skipping JSON file {file_path} due to unsupported format.")
            return
        generate_embeddings(sentences)

def generate_embeddings(sentences):
    if sentences:
        embeddings = model.encode(sentences, batch_size=532, show_progress_bar=True)
        print(f"Embeddings shape: {embeddings.shape}")
        # Optionally, save the embeddings to a file or use them directly
        # For example: np.save('embeddings.npy', embeddings)

# Step 6: Start processing the directory
process_directory(directory)