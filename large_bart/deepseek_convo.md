
I need to modify the existing code to select the document with the highest entropy among similar documents.

First, I should understand what entropy means in this context. Entropy is a measure of uncertainty or randomness in a probability distribution. For a document, I can calculate the entropy based on the classification scores of the candidate labels.

I need to calculate the entropy for each document's classification scores. The formula for entropy is:

Entropy = -σ (p_i * log(p_i)) for i in all labels

I should write a function to compute the entropy given a list of scores.

Next, I need to group similar documents together. Similarity can be determined using cosine similarity between document embeddings.

I can compute the embedding for each document using the embedding model. Then, I can compare each pair of documents to find which ones are similar.

To avoid redundant comparisons, I can use a clustering approach or a threshold on cosine similarity to group similar documents.

After grouping similar documents, within each group, I will select the document with the highest entropy.

I need to modify the code to store document embeddings and their entropy values.

I should collect all documents first, compute their embeddings and entropy, then group similar documents, and finally select the one with the highest entropy in each group.

This might be computationally expensive if there are many documents, so I need to consider efficiency.

Perhaps I can use a threshold to decide if two documents are similar based on their cosine similarity.

I should also ensure that the selected documents are diverse in terms of their topics.

I need to integrate this logic after processing all documents, so I should collect all necessary data first.

I should modify the main function to collect all necessary information for all documents before selecting the ones with highest entropy among similar documents.

I need to make sure that the results are saved correctly after this selection process.

I should test the code with sample data to verify that it works as expected.

To modify the code so that for any two or more similar documents, the document with the higher entropy is selected, follow these steps:

Calculate Entropy for Each Document:

Create a function to compute the entropy of the classification scores.

Use the entropy to measure the uncertainty of the document's classification.

Compute Document Embeddings:

Generate embeddings for each document using the embedding model.

These embeddings will be used to calculate cosine similarity between documents.

Determine Document Similarity:

Define a similarity threshold (e.g., 0.8) to consider documents as similar.

Compare each pair of documents using cosine similarity.

Group similar documents together.

Select Documents with Highest Entropy:

For each group of similar documents, select the one with the highest entropy.

Collect these selected documents as the final output.

Update the Output:

Modify the CSV and JSON outputs to include only the selected documents.

Ensure the results reflect the highest quality documents based on entropy.

Modified Code:

python
Copy
import os
import csv
import json
import torch
import signal
import sys
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np

# ... (previous code remains unchanged up to the main() function)

def compute_entropy(scores):
    """Compute entropy of a list of scores."""
    scores = np.array(scores)
    probs = scores / scores.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return entropy

def main():
    global processed_files

    # ... (previous code remains the same)

    # Collect all documents and their data
    all_docs = []

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
                print(f"Error processing file '{file_name}': {e}")
        else:
            print(f"Skipping non-text file or already processed file: {file_name}")

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
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(selected_docs, f, indent=4, default=lambda x: str(x))

    # Save final state
    save_state(processed_files)

    print(f"\nSelected documents saved to '{output_csv}' and '{output_json}'.")
    print(f"Total selected documents: {len(selected_docs)}")

if __name__ == "__main__":
    main()
Explanation:

Entropy Calculation:

Added compute_entropy function to calculate the entropy of classification scores.

Collecting Document Data:

Instead of immediately saving results, collect all document data in all_docs list.

Grouping Similar Documents:

Compared each pair of documents using cosine similarity.

Grouped documents with similarity above the threshold (0.8).

Selecting Highest Entropy Document:

Within each group of similar documents, selected the one with the highest entropy.

Saving Selected Documents:

Overwrote the CSV and JSON outputs with only the selected documents.

Ensured the outputs reflect the highest quality documents based on entropy.

Notes:

Similarity Threshold:

Adjust the similarity threshold in if sim > 0.8: to suit your requirements.

Computational Efficiency:

Pairwise comparison of documents is O(N²), which may be slow for large N.

For large datasets, consider clustering algorithms to group similar documents more efficiently.

Entropy Interpretation:

Higher entropy indicates more uncertainty in classification.

Selecting documents with higher entropy may capture more diverse or ambiguous content.

Data Persistence:

The state is saved upon interruption and resumed from where it left off.

Ensure that the state file and processed files are managed correctly.

This modification ensures that among similar documents, the one with the highest entropy is selected, effectively choosing the highest quality document based on classification uncertainty.
