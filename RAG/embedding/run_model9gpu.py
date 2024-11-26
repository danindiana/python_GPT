import os
import json
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Choose a pre-trained model
model_name = "all-MiniLM-L6-v2"

# Step 2: Load the pre-trained SentenceTransformer model
# Utilize GPUs by setting device="cuda"
model = SentenceTransformer(model_name, device="cuda")

# Step 3: Check the number of GPUs available
logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")

# Step 4: Prompt the user for the directory
directory = input("Enter the directory containing text or JSON files: ")

# Step 5: Recursively process files in the directory
def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".txt"):
                    logging.info(f"Processing text file: {file_path}")
                    process_text_file(file_path)
                elif file.endswith(".json"):
                    logging.info(f"Processing JSON file: {file_path}")
                    process_json_file(file_path)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

def process_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
            sentences = text_content.split('\n')
            generate_embeddings(sentences)
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")

def process_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_content = json.load(file)
            # Assuming the JSON structure is a list of sentences or a dictionary with a 'text' key
            if isinstance(json_content, list):
                sentences = json_content
            elif isinstance(json_content, dict) and 'text' in json_content:
                sentences = json_content['text'].split('\n')
            else:
                logging.warning(f"Skipping JSON file {file_path} due to unsupported format.")
                return
            generate_embeddings(sentences)
    except Exception as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")

def generate_embeddings(sentences):
    if sentences:
        try:
            embeddings = model.encode(sentences, batch_size=2332, show_progress_bar=True)
            logging.info(f"Embeddings shape: {embeddings.shape}")
            save_embeddings(embeddings, 'embeddings.npy')
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")

def save_embeddings(embeddings, file_path):
    try:
        np.save(file_path, embeddings)
        logging.info(f"Embeddings saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving embeddings to {file_path}: {e}")

# Step 6: Start processing the directory
process_directory(directory)