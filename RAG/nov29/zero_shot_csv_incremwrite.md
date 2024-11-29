The large `bart` script you are running writes the **CSV file** to the specified output file path (e.g., `classification_results_with_similarity.csv`) **after processing all the files**. 

### **Key Points about Output Writing**

1. **File Writing Logic**:
   - The results are accumulated in a list called `results` as each file is processed:
     ```python
     results.append({
         "file_name": file_name,
         "labels": result["labels"],
         "scores": result["scores"],
         "cosine_similarities": cosine_similarities
     })
     ```
   - Once all files are processed, the `results` list is written to the CSV file in one go:
     ```python
     save_results_csv(results, output_csv)
     ```

2. **Timing of CSV Generation**:
   - The script does **not** write the CSV file incrementally.
   - The CSV file is created only **after all text files** in the directory have been processed.

3. **Output File Location**:
   - The CSV file is saved in the same directory as the script unless a specific path is provided for the `output_csv` variable:
     ```python
     output_csv = "classification_results_with_similarity.csv"
     ```

### **What Happens During Processing?**
- While the files are being processed:
  - The CSV file does not yet exist.
  - Progress and results are printed to the console for each file, but nothing is written to disk until processing is complete.

---

### **Advantages of This Approach**
- Writing the file after all files are processed ensures that:
  - You don't risk partial or corrupted files if the process is interrupted.
  - The output is neatly written in one operation, reducing the chance of file I/O errors.

---

### **What if You Want Incremental Writing?**

If you want the CSV file to be updated incrementally while processing, you can modify the script to write results to the file as they are generated.

1. **Update the `save_results_csv` Function**:
   Modify it to append rows to the CSV file during each iteration:
   ```python
   def save_results_csv_incrementally(result, output_file):
       """Append a single result to the CSV file."""
       with open(output_file, "a", newline="", encoding="utf-8") as f:
           writer = csv.writer(f)
           for label, score, cosine_sim in zip(result["labels"], result["scores"], result["cosine_similarities"]):
               writer.writerow([result["file_name"], label, score, cosine_sim])
   ```

2. **Initialize the CSV File with a Header**:
   At the start of the script, create the CSV file and write the header:
   ```python
   with open(output_csv, "w", newline="", encoding="utf-8") as f:
       writer = csv.writer(f)
       writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])
   ```

3. **Call the Incremental Function in the Processing Loop**:
   Replace the `results.append(...)` logic with:
   ```python
   save_results_csv_incrementally({
       "file_name": file_name,
       "labels": result["labels"],
       "scores": result["scores"],
       "cosine_similarities": cosine_similarities
   }, output_csv)
   ```

---

### **Pros and Cons of Incremental Writing**

#### **Pros**:
- Partial results are available immediately if the process is interrupted.
- You can monitor progress by inspecting the CSV file as the script runs.

#### **Cons**:
- Slightly slower overall due to frequent file I/O operations.
- File integrity issues could arise if the process is interrupted during a write operation.

---

### **Summary**
The original script writes the CSV file only **after all files are processed**, ensuring a clean and atomic output. If you prefer incremental updates for progress tracking or fault tolerance, you can modify the script as outlined above. Let me know if you'd like help implementing this! 😊

Sure, I can help you modify the script to update the CSV file incrementally while processing. Here is the updated script with the necessary changes:

```python
import os
import csv
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModel

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

    # Load embedding model for cosine similarity
    print("Loading embedding model for cosine similarity...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    print("\nProcessing text files...\n")

    results = []  # Store results for output

    # Initialize the CSV file with a header
    output_csv = "classification_results_with_similarity.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])

    # Process each text file in the directory
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)

        # Only process text files
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
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

                # Print results to the console
                print(f"File: {file_name}")
                print("Labels, scores, and cosine similarities:")
                for label, score, cosine_sim in zip(result["labels"], result["scores"], cosine_similarities):
                    print(f"  - {label}: score={score:.4f}, cosine_similarity={cosine_sim:.4f}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")
        else:
            print(f"Skipping non-text file: {file_name}")

    # Save results to JSON
    output_json = "classification_results_with_similarity.json"
    save_results_json(results, output_json)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")

if __name__ == "__main__":
    main()
```

In this updated script:
1. The `save_results_csv_incrementally` function is defined to append a single result to the CSV file.
2. The CSV file is initialized with a header at the start of the script.
3. The `save_results_csv_incrementally` function is called within the processing loop to save results incrementally.

This ensures that the CSV file is updated in real-time as each file is processed.
