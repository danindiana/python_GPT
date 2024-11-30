import os
import csv
import json
import torch
import signal
import sys
from transformers import pipeline, AutoTokenizer, AutoModel

# Global variables to keep track of state
state_file = "processing_state.json"
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

    results = []  # Store results for output

    # Initialize the CSV file with a header
    output_csv = "classification_results_with_similarity.csv"
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])

    # Process each text file in the directory
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)

        # Only process text files
        if os.path.isfile(file_path) and file_path.endswith(".txt") and file_name not in processed_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:  # Skip empty or invalid files
                print(f"Skipping empty file: {file_name}")
                continue

            # Perform zero-shot classification
            try:
                result = classifier(text, candidate_labels=labels, multi_label=True)

                # Compute cosine similarity
                text_embedding = get_text_embedding(text, tokenizer, embedding_model)
                label_embeddings = [
                    get_text_embedding(label, tokenizer, embedding_model)
                    for label in labels
                ]
                cosine_similarities = [compute_cosine_similarity(text_embedding, label_embedding) for label_embedding in label_embeddings]

                # Collect result
                results.append({
                    "file_name": file_name,
                    "labels": result["labels"],
                    "scores": result["scores"],
                    "cosine_similarities": cosine_similarities
                })

                # Save result incrementally to CSV
                save_results_csv_incrementally({
                    "file_name": file_name,
                    "labels": result["labels"],
                    "scores": result["scores"],
                    "cosine_similarities": cosine_similarities
                }, output_csv)

                # Mark file as processed
                processed_files.add(file_name)

                # Print results to the console
                print(f"File: {file_name}")
                print("Labels, scores, and cosine similarities:")
                for label, score, cosine_sim in zip(result["labels"], result["scores"], cosine_similarities):
                    print(f"  - {label}: score={score:.4f}, cosine_similarity={cosine_sim:.4f}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")
        else:
            print(f"Skipping non-text file or already processed file: {file_name}")

    # Save results to JSON
    output_json = "classification_results_with_similarity.json"
    save_results_json(results, output_json)

    # Save final state
    save_state(processed_files)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")

if __name__ == "__main__":
    main()
