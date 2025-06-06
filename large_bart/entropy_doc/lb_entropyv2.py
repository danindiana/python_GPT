import os
import csv
import json
import torch
import signal
import sys
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset

# Global variables to keep track of state
output_dir = "/home/jeb/programs/largebart/sus_resum/output"  # Specify the output directory

# Check if the directory exists and has write permissions
if not os.path.exists(output_dir):
    print(f"Error: Directory '{output_dir}' does not exist.")
    sys.exit(1)

if not os.access(output_dir, os.W_OK):
    print(f"Error: You do not have write permissions for the directory '{output_dir}'.")
    sys.exit(1)

state_file = os.path.join(output_dir, "processing_state.json")
output_csv = os.path.join(output_dir, "classification_results_with_similarity.csv")
output_json = os.path.join(output_dir, "classification_results_with_similarity.json")
processed_files = set()

def list_available_devices():
    """List available GPU devices and return the selected device."""
    if not torch.cuda.is_available():
        print("No GPU devices available. Running on CPU.")
        return -1  # Use CPU

    print("Available GPU devices:")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    while True:
        device_id = input("Select a GPU device ID (or press Enter to use CPU): ").strip()
        if device_id == "":
            print("Using CPU.")
            return -1
        if device_id.isdigit() and int(device_id) < torch.cuda.device_count():
            print(f"Using GPU device {device_id}: {torch.cuda.get_device_name(int(device_id))}")
            return int(device_id)
        else:
            print("Invalid selection. Please enter a valid GPU device ID.")

def compute_cosine_similarity(vector1, vector2):
    """Compute cosine similarity between two vectors."""
    vector1 = torch.tensor(vector1)
    vector2 = torch.tensor(vector2)
    return torch.nn.functional.cosine_similarity(vector1, vector2, dim=0).item()

def get_text_embedding(text, tokenizer, model):
    """Generate an embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
    # Handle cases where the tensor may not have sufficient dimensions
    if len(outputs.shape) > 1:
        return outputs.mean(dim=1).squeeze().tolist()  # Average pooling
    else:
        return outputs.squeeze().tolist()  # Handle single-dimension tensors

def compute_entropy(scores):
    """Compute entropy of a list of scores."""
    scores = np.array(scores)
    probs = scores / scores.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return entropy

def save_results_csv_incrementally(result, output_file):
    """Append a single result to the CSV file."""
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for label, score, cosine_sim in zip(result["labels"], result["scores"], result["cosine_similarities"]):
            writer.writerow([result["file_name"], label, score, cosine_sim])

def save_results_json(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

def save_state(processed_files):
    """Save the state of processed files to a JSON file."""
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=4)

def load_state():
    """Load the state of processed files from a JSON file."""
    global processed_files
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            processed_files = set(json.load(f))

def signal_handler(sig, frame):
    """Handle interrupt signals and save state before exiting."""
    print("\nInterrupt received, saving state and exiting...")
    save_state(processed_files)
    sys.exit(0)

def main():
    global processed_files

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load existing state
    load_state()

    # Prompt user for the directory path containing text files
    target_dir = input("Enter the path to the directory containing text files: ").strip()

    # Verify the directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # Prompt user for target labels (comma-separated)
    labels_input = input("Enter target labels (comma-separated, e.g., 'science, law, biology, technology'): ").strip()
    labels = [label.strip() for label in labels_input.split(",")]

    if not labels:
        print("Error: No labels provided.")
        return

    # Select device (CPU or GPU)
    device = list_available_devices()

    # Load the zero-shot classification pipeline
    print("Loading facebook/bart-large-mnli model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    # Load embedding model for cosine similarity
    print("Loading embedding model for cosine similarity...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    print("\nProcessing text files...\n")

    # Load text files into a dataset
    def generate_examples():
        for file_name in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file_name)
            if os.path.isfile(file_path) and file_path.endswith(".txt") and file_name not in processed_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    yield {"file_name": file_name, "text": text}

    dataset = Dataset.from_generator(generate_examples)

    all_docs = []  # Collect all documents and their data

    # Initialize the CSV file with a header
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])

    # Process the dataset in batches
    batch_size = 16  # Adjust batch size as needed
    for start_idx in range(0, len(dataset), batch_size):
        batch = dataset[start_idx:start_idx + batch_size]
        texts = batch["text"]
        file_names = batch["file_name"]

        # Perform zero-shot classification in batch
        try:
            results = classifier(texts, candidate_labels=labels, multi_label=True)

            for i, result in enumerate(results):
                text = texts[i]
                file_name = file_names[i]

                # Compute cosine similarity
                text_embedding = get_text_embedding(text, tokenizer, embedding_model)
                label_embeddings = [
                    get_text_embedding(label, tokenizer, embedding_model)
                    for label in labels
                ]
                cosine_similarities = [compute_cosine_similarity(text_embedding, label_embedding) for label_embedding in label_embeddings]

                # Compute entropy
                entropy = compute_entropy(result["scores"])

                # Collect document data
                all_docs.append({
                    "file_name": file_name,
                    "text": text,
                    "embedding": text_embedding,
                    "entropy": entropy,
                    "labels": result["labels"],
                    "scores": result["scores"],
                    "cosine_similarities": cosine_similarities
                })

                # Mark file as processed
                processed_files.add(file_name)

                print(f"Processed file: {file_name}")
        except Exception as e:
            print(f"Error processing batch starting at index {start_idx}: {e}")

    # Now, group similar documents and select the one with highest entropy in each group
    selected_docs = []
    used_indices = set()

    # Compare each pair of documents
    for i in range(len(all_docs)):
        if i in used_indices:
            continue
        # Group similar documents
        group = [i]
        for j in range(i + 1, len(all_docs)):
            sim = compute_cosine_similarity(torch.tensor(all_docs[i]['embedding']), torch.tensor(all_docs[j]['embedding']))
            if sim > 0.8:  # Adjust the similarity threshold as needed
                group.append(j)
                used_indices.add(j)
        # Select the document with highest entropy in the group
        max_entropy_idx = group[0]
        for idx in group:
            if all_docs[idx]['entropy'] > all_docs[max_entropy_idx]['entropy']:
                max_entropy_idx = idx
        selected_docs.append(all_docs[max_entropy_idx])

    # Save the selected documents to CSV and JSON
    if os.path.exists(output_csv):
        os.remove(output_csv)  # Clear previous data
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])
        for doc in selected_docs:
            for label, score, cosine_sim in zip(doc["labels"], doc["scores"], doc["cosine_similarities"]):
                writer.writerow([doc["file_name"], label, score, cosine_sim])

    # Save results to JSON
    save_results_json(selected_docs, output_json)

    # Save final state
    save_state(processed_files)

    print(f"\nSelected documents saved to '{output_csv}' and '{output_json}'.")
    print(f"Total selected documents: {len(selected_docs)}")

if __name__ == "__main__":
    main()
