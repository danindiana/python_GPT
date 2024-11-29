To assign probabilities to each document being in a given domain (e.g., science, law, biology) using the **`facebook/bart-large-mnli`** model for zero-shot classification, you can follow this step-by-step approach. This model supports "zero-shot" classification by providing domain-specific labels or hypotheses without needing fine-tuning.

---

### **Step-by-Step Guide**

#### **1. Install Dependencies**
Ensure you have the Hugging Face Transformers library installed:
```bash
pip install transformers
```

#### **2. Load the Model and Tokenizer**
Load the **`facebook/bart-large-mnli`** model from the Hugging Face Transformers library:
```python
from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
```

---

#### **3. Define Your Labels (Domains)**
Define the list of target labels representing the domains of interest. For example:
```python
labels = ["science", "law", "biology", "computer science", "literature"]
```

---

#### **4. Classify a Single Document**
Pass the document and your labels to the classifier. You can also specify the hypothesis format used for classification:
```python
document = """
Quantum computing is a rapidly evolving field of science that leverages the principles of quantum mechanics
to perform computations far more efficiently than classical computers in certain domains.
"""

result = classifier(
    document,
    candidate_labels=labels,
    multi_label=True  # Allows overlapping probabilities for multiple domains
)

# Display the classification results
print(result)
```

---

#### **5. Interpret the Results**
The result will include:
- `labels`: The list of domain labels sorted by their likelihood.
- `scores`: Probabilities (or confidence scores) for each domain.

Example output:
```python
{
    'sequence': 'Quantum computing is a rapidly evolving field...',
    'labels': ['science', 'computer science', 'biology', 'law', 'literature'],
    'scores': [0.84, 0.72, 0.15, 0.05, 0.03]
}
```

---

#### **6. Process a Batch of Documents**
For multiple documents, loop through your dataset and classify each:
```python
documents = [
    "Quantum mechanics forms the basis of modern physics.",
    "The Supreme Court ruling set a new precedent in constitutional law.",
    "Artificial intelligence is transforming computer science and technology.",
    "The study of genetics has revolutionized modern biology."
]

# Classify each document
for doc in documents:
    result = classifier(doc, candidate_labels=labels, multi_label=True)
    print(f"Document: {doc[:50]}...")  # Print a snippet of the document
    print(f"Labels: {result['labels']}")
    print(f"Scores: {result['scores']}\n")
```

---

#### **7. Filter Based on Probabilities**
If you want to include only documents that meet a relevance threshold, filter them by score:
```python
threshold = 0.5  # Set a threshold for relevance

filtered_documents = []
for doc in documents:
    result = classifier(doc, candidate_labels=labels, multi_label=True)
    relevant_labels = [
        (label, score) for label, score in zip(result["labels"], result["scores"])
        if score >= threshold
    ]
    if relevant_labels:
        filtered_documents.append({"document": doc, "relevant_labels": relevant_labels})

# Print filtered documents
for item in filtered_documents:
    print(f"Document: {item['document'][:50]}...")
    print(f"Relevant Labels: {item['relevant_labels']}\n")
```

---

#### **8. (Optional) Save Results**
Save the classification results and scores for further processing:
```python
import pandas as pd

results = []
for doc in documents:
    result = classifier(doc, candidate_labels=labels, multi_label=True)
    results.append({"document": doc, "labels": result["labels"], "scores": result["scores"]})

# Convert to a DataFrame and save
df = pd.DataFrame(results)
df.to_csv("classified_documents.csv", index=False)
```

---

### **Advantages of `facebook/bart-large-mnli`**
- **Zero-shot capability**: No training required; just specify your target labels.
- **Multi-label classification**: Allows overlapping relevance for multiple domains.
- **Flexible hypotheses**: You can use custom phrases as labels (e.g., "related to computer science").

If you'd like detailed guidance on improving accuracy (e.g., customizing label descriptions or preprocessing documents), let me know!
