import os
import csv
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from datasets import Dataset  # Hugging Face Dataset API

def list_available_devices():
    """List available GPU devices and return the selected device ID and torch device."""
    if not torch.cuda.is_available():
        print("No GPU devices available. Running on CPU.")
        return -1, torch.device("cpu")  # Use CPU

    print("Available GPU devices:")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    while True:
        device_id = input("Select a GPU device ID (or press Enter to use CPU): ").strip()
        if device_id == "":
            print("Using CPU.")
            return -1, torch.device("cpu")
        if device_id.isdigit() and int(device_id) < torch.cuda.device_count():
            selected_id = int(device_id)
            print(f"Using GPU device {selected_id}: {torch.cuda.get_device_name(selected_id)}")
            return selected_id, torch.device(f"cuda:{selected_id}")
        else:
            print("Invalid selection. Please enter a valid GPU device ID.")

def compute_cosine_similarity(vector1, vector2):
    """Compute cosine similarity between two vectors."""
    vector1 = torch.tensor(vector1)
    vector2 = torch.tensor(vector2)
    return torch.nn.functional.cosine_similarity(vector1, vector2, dim=0).item()

def get_text_embedding(text, tokenizer, model, torch_device):
    """Generate an embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(torch_device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
    if len(outputs.shape) > 1:
        return outputs.mean(dim=1).squeeze().tolist()  # Average pooling
    else:
        return outputs.squeeze().tolist()  # Handle single-dimension tensors

def save_results_csv(results, output_file):
    """Save results to a CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])  # Header
        for result in results:
            for label, score, cosine_sim in zip(result["labels"], result["scores"], result["cosine_similarities"]):
                writer.writerow([result["file_name"], label, score, cosine_sim])

def save_results_json(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

def clean_text(text):
    """Remove or replace invalid UTF-8 characters."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'ignore')
    else:
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text

def main():
    # Prompt user for the directory path containing text files
    target_dir = input("Enter the path to the directory containing text files: ").strip()

    # Verify the directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # Prompt user for target labels (comma-separated)
    labels_input = input("Enter target labels (comma-separated, e.g., 'science, law, biology, technology'): ").strip()
    labels = [label.strip() for label in labels_input.split(",") if label.strip()]

    if not labels:
        print("Error: No labels provided.")
        return

    # Select device (CPU or GPU)
    device_id, torch_device = list_available_devices()

    # Load the zero-shot classification pipeline
    print("Loading facebook/bart-large-mnli model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)

    # Load embedding model for cosine similarity
    print("Loading embedding model for cosine similarity...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(torch_device)

    print("\nProcessing text files...\n")

    # Create a Hugging Face Dataset from the text files
    dataset_dict = {"file_name": [], "text": []}
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(".txt"):
            try:
                # Open the file with 'ignore' mode to skip invalid UTF-8 characters
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                text = clean_text(text)  # Clean the text
                if text:  # Skip empty files
                    dataset_dict["file_name"].append(file_name)
                    dataset_dict["text"].append(text)
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")

    if not dataset_dict["file_name"]:
        print("No valid text files found to process.")
        return

    dataset = Dataset.from_dict(dataset_dict)

    # Process the dataset in batches
    results = []
    batch_size = 10  # Adjust batch size as needed

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")

        file_names = batch["file_name"]
        texts = batch["text"]

        # Perform zero-shot classification in batch
        try:
            batch_results = classifier(texts, candidate_labels=labels, multi_label=True)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            # Attempt to process files individually in the failed batch
            for j, (file_name, text) in enumerate(zip(file_names, texts)):
                print(f"  Attempting to process file '{file_name}' individually...")
                try:
                    individual_result = classifier(text, candidate_labels=labels, multi_label=True)
                    
                    text_embedding = get_text_embedding(text, tokenizer, embedding_model, torch_device)
                    label_embeddings = [get_text_embedding(label, tokenizer, embedding_model, torch_device) for label in labels]
                    cosine_similarities = [compute_cosine_similarity(text_embedding, label_embedding) for label_embedding in label_embeddings]

                    results.append({
                        "file_name": file_name,
                        "labels": individual_result["labels"],
                        "scores": individual_result["scores"],
                        "cosine_similarities": cosine_similarities
                    })

                    # Print results to the console
                    print(f"File: {file_name}")
                    print("Labels, scores, and cosine similarities:")
                    for label, score, cosine_sim in zip(individual_result["labels"], individual_result["scores"], cosine_similarities):
                        print(f"  - {label}: score={score:.4f}, cosine_similarity={cosine_sim:.4f}")
                    print("-" * 50)
                except Exception as e_individual:
                    print(f"    Error processing file '{file_name}': {e_individual}")
            continue  # Move to the next batch after handling the current failed batch

        # If batch processing succeeded, process each file in the batch
        for file_name, text, result in zip(file_names, texts, batch_results):
            try:
                text_embedding = get_text_embedding(text, tokenizer, embedding_model, torch_device)
                label_embeddings = [get_text_embedding(label, tokenizer, embedding_model, torch_device) for label in labels]
                cosine_similarities = [compute_cosine_similarity(text_embedding, label_embedding) for label_embedding in label_embeddings]

                # Collect result
                results.append({
                    "file_name": file_name,
                    "labels": result["labels"],
                    "scores": result["scores"],
                    "cosine_similarities": cosine_similarities
                })

                # Print results to the console
                print(f"File: {file_name}")
                print("Labels, scores, and cosine similarities:")
                for label, score, cosine_sim in zip(result["labels"], result["scores"], cosine_similarities):
                    print(f"  - {label}: score={score:.4f}, cosine_similarity={cosine_sim:.4f}")
                print("-" * 50)
            except Exception as e_file:
                print(f"Error processing file '{file_name}': {e_file}")
                continue

    # Save results to CSV and JSON
    output_csv = "classification_results_with_similarity.csv"
    output_json = "classification_results_with_similarity.json"
    save_results_csv(results, output_csv)
    save_results_json(results, output_json)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")

if __name__ == "__main__":
    main()
