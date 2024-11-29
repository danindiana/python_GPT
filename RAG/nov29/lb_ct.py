import os
import torch
from transformers import pipeline

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

def main():
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

    print("\nProcessing text files...\n")
    
    # Process each text file in the directory
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)

        # Only process text files
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Perform zero-shot classification
            try:
                result = classifier(text, candidate_labels=labels, multi_label=True)
                
                # Print results to the console
                print(f"File: {file_name}")
                print("Labels and scores:")
                for label, score in zip(result["labels"], result["scores"]):
                    print(f"  - {label}: {score:.4f}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")
        else:
            print(f"Skipping non-text file: {file_name}")

if __name__ == "__main__":
    main()
