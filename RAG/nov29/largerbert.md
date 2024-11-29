To clean and sort text files based on topicality for inclusion in a Retrieval-Augmented Generation (RAG) vector database and a local language model ensemble, you can employ a multi-step pipeline using the following tools and techniques. Here's an outline of the process:

---

### **1. Preprocessing the Text**
- **OCR Post-Processing**: If the text comes from OCR, clean it with the following tools:
  - **`pytesseract`** for OCR (ensure high-quality extraction).
  - **Regular Expressions (`re`)**: Clean noisy data, remove unwanted characters, and normalize whitespace.
  - **`langdetect` or `langid`**: Detect and filter out non-relevant languages.
  - **`unidecode`**: Normalize text for consistent processing.

---

### **2. Text Cleaning**
- **Stopword Removal**: Use **NLTK**, **spaCy**, or **Gensim** to remove common stopwords.
- **Custom Filters**: Define domain-specific keywords or phrases to exclude irrelevant files (e.g., dictionaries of terms for science, computer science, etc.).
- **Deduplication**: Use tools like **MinHash** or **SimHash** (via `datasketch`) to remove duplicate files.

---

### **3. Semantic Filtering Based on Domain**
#### **a. Topic Modeling**
- **LDA (Latent Dirichlet Allocation)**:
  - Libraries: **Gensim**, **spaCy**, **Scikit-learn**.
  - Use topic modeling to cluster and identify the major topics in your text corpus.
- **BERT Embeddings**:
  - Pre-trained models like **`sentence-transformers`** (e.g., `all-MiniLM-L6-v2`) to encode documents into embeddings.
  - Use **UMAP** or **t-SNE** for dimensionality reduction to identify clusters visually.

#### **b. Pre-trained Text Classifiers**
- Fine-tune or use pre-trained classifiers with libraries such as:
  - **Hugging Face Transformers**: Fine-tune models like **BERT**, **DistilBERT**, or **GPT** for domain-specific text classification.
  - **Zero-shot Classification**: Models like **`facebook/bart-large-mnli`** to classify text without needing labeled training data.
  - Example: Assign probabilities to each document being in a given domain (e.g., science, law, biology).

#### **c. Keywords & Regular Expressions**
- Use **TF-IDF** or **custom keyword dictionaries**:
  - Extract top terms from a sample corpus for each domain.
  - Match documents with a threshold score based on relevant keywords.

---

### **4. Clustering for Relevance**
- **Text Embeddings**: Generate embeddings for each document using:
  - Models from **`sentence-transformers`**.
  - GPU-accelerated libraries like **RAPIDS cuML** for scalable clustering.
- **Clustering Algorithms**:
  - **KMeans**: Partition documents based on similarity.
  - **HDBSCAN**: Identify natural clusters in data without predefining the number of clusters.
- **Relevance Scoring**:
  - Use **Cosine Similarity** or **Euclidean Distance** to rank documents' relevance to your target domains.

---

### **5. Building the RAG Pipeline**
#### **a. Vector Database**
- Use a local vector database to store embeddings for quick retrieval:
  - **FAISS**: Facebook's efficient similarity search.
  - **Weaviate**, **Milvus**, or **Pinecone** (all open-source/local deployable options).

#### **b. Metadata Tags**
- During the ingestion phase, tag documents with metadata:
  - Domain: Science, Law, Biology, etc.
  - Keywords: Extracted terms that link to the domain.
  - Quality: Include a relevancy score.

#### **c. Local LLM Ensemble**
- Pair the vector database with a local ensemble of fine-tuned LLMs:
  - Fine-tune models like **LLaMA** or **GPT-J** for retrieval-based tasks.
  - Use **RAG frameworks** like Haystack or LangChain to combine retrieval with generative capabilities.

---

### **6. Automation and Scaling**
- **Snakemake** or **Airflow**: Automate and scale the pipeline.
- **Batch Processing**:
  - Process text files in parallel with tools like **Dask** or **Spark**.
- **Monitoring**:
  - Log and monitor the sorting and cleaning process using **MLFlow** or **Weights & Biases**.

---

### Example Workflow
1. Preprocess all text files (cleaning and deduplication).
2. Classify files into domains using zero-shot classification or fine-tuned models.
3. Cluster text files based on embeddings and filter by relevance.
4. Store cleaned and classified documents in a vector database with metadata.
5. Integrate the database with an LLM-based retrieval system for downstream tasks.

Would you like detailed examples or code snippets for any specific step?

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

Choosing between generating a large list of **target labels** (domains/topics you want) or labels for **things you don’t want** depends on the scope and specificity of your task. Here's how to approach this problem:

---

### **Option 1: Generate a Large List of Target Labels**

#### **Advantages**
- Ensures specificity by capturing a wide variety of desirable topics.
- Supports a more focused downstream task since the model has explicit instructions about what is relevant.

#### **How to Generate a Large List of Target Labels**
1. **Start with High-Level Categories**:
   Begin with broad categories like "science," "law," "technology," and refine them into subdomains:
   - **Science**: Biology, Physics, Chemistry, Environmental Science, Neuroscience.
   - **Technology**: Artificial Intelligence, Machine Learning, Computer Science, Software Engineering.
   - **Law**: Constitutional Law, Contract Law, Intellectual Property, Human Rights Law.

2. **Expand Using Resources**:
   - **Wikipedia or Knowledge Graphs**:
     - Use category hierarchies from Wikipedia (e.g., https://en.wikipedia.org/wiki/Category:Science).
     - Scrape or query knowledge bases like **Wikidata** for related topics.
   - **Ontology Databases**:
     - Explore ontologies like **WordNet**, **DBpedia**, or **ConceptNet** to find related terms.
   - **Academic Taxonomies**:
     - Use taxonomies from academic journals or libraries (e.g., ACM Classification System, MeSH for medical terms).
   - **AI Tools**:
     - Use GPT or other large language models to generate hierarchical lists:
       ```python
       from transformers import pipeline
       
       generator = pipeline("text-generation", model="gpt-3.5-turbo")
       prompt = "Generate a list of subdomains for science, technology, and law:"
       response = generator(prompt, max_length=200)
       print(response[0]["generated_text"])
       ```

3. **Curate and Organize**:
   - Cluster the generated labels into categories using **manual review** or **unsupervised clustering** (e.g., KMeans with embeddings).
   - Save them in a structured format (e.g., JSON or CSV).

---

### **Option 2: Work in Reverse and Generate a List of Labels for Exclusion**

#### **Advantages**
- Easier for tasks with a broad focus where only a few topics need exclusion.
- Simpler to maintain when the range of relevant topics is naturally wide.

#### **How to Generate an Exclusion List**
1. **Identify Irrelevant Categories**:
   - Start with obvious exclusions like "entertainment," "fashion," "sports," etc.
   - Use resources like Wikipedia or GPT to generate lists of unrelated domains:
     ```python
     from transformers import pipeline

     generator = pipeline("text-generation", model="gpt-3.5-turbo")
     prompt = "Generate a list of unrelated categories for a corpus focused on science, technology, and law:"
     response = generator(prompt, max_length=200)
     print(response[0]["generated_text"])
     ```

2. **Fine-Tune Based on Your Data**:
   - Scan a sample of your dataset to identify common irrelevant topics.
   - Add specific labels to the exclusion list dynamically (e.g., documents with frequent mentions of "celebrity," "sports," or "gossip").

3. **Automate Filtering**:
   - Use word frequencies or simple heuristics to detect irrelevant topics early.
   - Filter out documents matching terms from your exclusion list.

---

### **Option 3: Hybrid Approach**

Use a hybrid approach to balance the advantages of both:
1. Generate a **broad list of desirable target labels** as a baseline.
2. Augment it with an **exclusion list** to catch edge cases and filter irrelevant topics.
3. Continuously refine both lists by analyzing misclassified or borderline documents during the pipeline.

---

### **Practical Implementation**

1. **Use Pre-Trained Models with Flexible Labels**:
   With models like **`facebook/bart-large-mnli`**, you don’t need an exhaustive list of labels. Instead, provide a representative set of categories and let the model infer relevance.

2. **Leverage Embedding Similarity**:
   - Generate embeddings for your labels and documents.
   - Use a similarity threshold to determine relevance for documents even if their topics aren't explicitly labeled.

3. **Iterative Refinement**:
   - Start with a high-level list of desirable topics.
   - Analyze misclassifications or false positives/negatives to improve your target and exclusion lists.

---

### **Tools for Label Generation and Refinement**
- **GPT Models**:
  Generate hierarchical or domain-specific lists quickly.
- **Knowledge Bases**:
  Use **Wikidata**, **DBpedia**, or academic taxonomies for structured data.
- **Clustering Algorithms**:
  Cluster embeddings of generated labels or sample documents to discover new categories.
- **Human-in-the-Loop**:
  Incorporate feedback loops to refine lists with minimal manual effort.

---

Would you like help generating specific labels, an exclusion list, or automating this process?

Here is a Python script for **zero-shot classification** using the `facebook/bart-large-mnli` model. The script will prompt the user for a directory containing text files, classify each file against a list of user-provided target labels, and output the results to the console.

---

### Script: Zero-Shot Classification with `facebook/bart-large-mnli`

```python
import os
from transformers import pipeline

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

    # Load the zero-shot classification pipeline
    print("Loading facebook/bart-large-mnli model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

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
```

---

### **Instructions**

1. **Install Required Libraries**:
   Ensure you have the Hugging Face Transformers library installed. Run:
   ```bash
   pip install transformers
   ```

2. **Prepare Your Environment**:
   - Your system has CUDA 12.7, so ensure that **PyTorch** is installed with GPU support.
   - Install PyTorch with CUDA support:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu127
     ```

3. **Run the Script**:
   - Save the script as `zero_shot_classifier.py`.
   - Run the script:
     ```bash
     python3 zero_shot_classifier.py
     ```

4. **Provide Input**:
   - Enter the directory containing text files (e.g., `/path/to/text/files`).
   - Provide the target labels, such as `science, law, biology, technology`.

5. **View Output**:
   - The script processes each `.txt` file in the directory.
   - It outputs the classification results (labels and their probabilities) to the console.

---

### **Expected Console Output**
Example output when processing a file:

```plaintext
Enter the path to the directory containing text files: /path/to/text/files
Enter target labels (comma-separated, e.g., 'science, law, biology, technology'): science, law, biology, technology
Loading facebook/bart-large-mnli model...

Processing text files...

File: example1.txt
Labels and scores:
  - science: 0.8943
  - technology: 0.7321
  - biology: 0.4512
  - law: 0.1523
--------------------------------------------------
File: example2.txt
Labels and scores:
  - law: 0.8712
  - science: 0.3204
  - technology: 0.1021
  - biology: 0.0723
--------------------------------------------------
```

---

### **GPU Utilization**

- **Verify GPU Usage**:
  After running the script, check GPU utilization with:
  ```bash
  nvidia-smi
  ```
  Look for activity under the process using `python3`.

- The script will automatically utilize GPUs if PyTorch is installed with CUDA support and your drivers are configured correctly.

If you encounter issues or need further customization (e.g., adding logging, saving results, or improving performance), let me know!

You can modify the script to query the available GPU devices and prompt the user to select which GPU they wish to use. The `transformers` library supports specifying a `device` argument for the pipeline to leverage GPU acceleration.

Here’s how you can enhance the script to allow the user to choose a GPU device:

---

### **Modified Script with GPU Selection**
```python
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
```

---

### **Explanation of Changes**

1. **Check for Available GPUs**:
   The function `list_available_devices()` checks for available GPUs using `torch.cuda`.

2. **User Prompt for Device Selection**:
   - Lists all available GPU devices and their names.
   - Allows the user to select a GPU by entering its device ID.
   - Defaults to CPU if the user presses Enter or selects an invalid option.

3. **Set the Pipeline `device`**:
   - The `device` parameter in the `pipeline` function specifies whether to run on CPU (`-1`) or a specific GPU (`0`, `1`, etc.).

4. **Fallback to CPU**:
   - If no GPUs are available or the user chooses not to use one, the script will run on the CPU.

---

### **Execution Example**
- If GPUs are available:
  ```plaintext
  Available GPU devices:
    [0] NVIDIA GeForce RTX 3080
    [1] NVIDIA GeForce RTX 3060
  Select a GPU device ID (or press Enter to use CPU): 1
  Using GPU device 1: NVIDIA GeForce RTX 3060
  ```
- If the user presses Enter:
  ```plaintext
  Using CPU.
  ```

---

### **Verify GPU Utilization**
- Check GPU utilization during execution:
  ```bash
  nvidia-smi
  ```

This will confirm that the pipeline is utilizing the selected GPU.

Let me know if you need further customizations!

The current version of the classifier program **only prints the output to the console**. If you'd like to save the results to a file for later analysis, we can enhance the script to write the output to a structured format such as **CSV** or **JSON**.

Here’s how we can modify the script to include file output:

---

### **Output Options**
1. **Write to a CSV File**:
   - CSV is suitable for tabular data and easy to analyze with tools like Excel, Pandas, or databases.
   - Each row will include the file name, target labels, and their respective scores.

2. **Write to a JSON File**:
   - JSON is more flexible and suitable for hierarchical data.
   - Useful if you want to store detailed information about each file.

---

### **Enhanced Script with File Output**
Below is the modified script to save the output as both CSV and JSON files:

```python
import os
import torch
import csv
import json
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

def save_results_csv(results, output_file):
    """Save results to a CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label", "Score"])  # Header
        for result in results:
            for label, score in zip(result["labels"], result["scores"]):
                writer.writerow([result["file_name"], label, score])

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

    print("\nProcessing text files...\n")
    
    results = []  # Store results for output

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
                
                # Collect result
                results.append({
                    "file_name": file_name,
                    "labels": result["labels"],
                    "scores": result["scores"]
                })

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

    # Save results to CSV and JSON
    output_csv = "classification_results.csv"
    output_json = "classification_results.json"
    save_results_csv(results, output_csv)
    save_results_json(results, output_json)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")

if __name__ == "__main__":
    main()
```

---

### **Details of the File Outputs**
1. **CSV File (`classification_results.csv`)**:
   - Example content:
     ```csv
     File Name,Label,Score
     example1.txt,science,0.8943
     example1.txt,technology,0.7321
     example2.txt,law,0.8712
     example2.txt,biology,0.0723
     ```

2. **JSON File (`classification_results.json`)**:
   - Example content:
     ```json
     [
         {
             "file_name": "example1.txt",
             "labels": ["science", "technology", "biology"],
             "scores": [0.8943, 0.7321, 0.4512]
         },
         {
             "file_name": "example2.txt",
             "labels": ["law", "science", "technology"],
             "scores": [0.8712, 0.3204, 0.1021]
         }
     ]
     ```

---

### **How to Use**
1. **Run the Script**:
   Follow the same instructions as before to execute the script.

2. **Locate the Output**:
   After processing, you will find:
   - A `classification_results.csv` file for easy viewing in spreadsheets.
   - A `classification_results.json` file for hierarchical data storage.

3. **Analyze the Results**:
   - Use Python's `pandas` or any spreadsheet tool to further analyze the CSV.
   - Use the JSON for integration with other systems or applications.

---

Let me know if you want additional features like filtering, custom output filenames, or direct integration with databases!

The provided script generates **one large file** for all processed files in the target directory, combining the data into a single `CSV` and a single `JSON` file. This approach is already aligned with your preference for a consolidated output file.

Now, let’s enhance the script further to **add cosine similarity scores** for each document. The cosine similarity will indicate how closely the document aligns with the provided target labels based on text embeddings. Here's the updated script:

---

### **Enhanced Script with Cosine Similarity**
This script calculates the cosine similarity between the text embedding of each document and the text embeddings of the labels, and includes this score in the output.

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
    return torch.nn.functional.cosine_similarity(vector1, vector2).item()

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

                # Compute cosine similarity
                text_embedding = embedding_model(**tokenizer(text, return_tensors="pt", truncation=True, padding=True)).last_hidden_state.mean(dim=1).squeeze().tolist()
                label_embeddings = [
                    embedding_model(**tokenizer(label, return_tensors="pt", truncation=True, padding=True)).last_hidden_state.mean(dim=1).squeeze().tolist()
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

    # Save results to CSV and JSON
    output_csv = "classification_results_with_similarity.csv"
    output_json = "classification_results_with_similarity.json"
    save_results_csv(results, output_csv)
    save_results_json(results, output_json)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")

if __name__ == "__main__":
    main()
```

---

### **Key Enhancements**

1. **Cosine Similarity Calculation**:
   - Uses the `sentence-transformers/all-MiniLM-L6-v2` model to generate text embeddings.
   - Computes cosine similarity between the document embedding and each label embedding.

2. **Output Structure**:
   - **CSV**:
     Includes the file name, label, classification score, and cosine similarity for each label.
   - **JSON**:
     Includes the same data in a structured hierarchical format for flexibility.

---

### **Output Examples**

#### **CSV File** (`classification_results_with_similarity.csv`):
```csv
File Name,Label,Score,Cosine Similarity
example1.txt,science,0.8943,0.9123
example1.txt,technology,0.7321,0.8412
example2.txt,law,0.8712,0.8123
example2.txt,biology,0.4512,0.7641
```

#### **JSON File** (`classification_results_with_similarity.json`):
```json
[
    {
        "file_name": "example1.txt",
        "labels": ["science", "technology"],
        "scores": [0.8943, 0.7321],
        "cosine_similarities": [0.9123, 0.8412]
    },
    {
        "file_name": "example2.txt",
        "labels": ["law", "biology"],
        "scores": [0.8712, 0.4512],
        "cosine_similarities": [0.8123, 0.7641]
    }
]
```

---

### **How to Use**

1. **Install Required Libraries**:
   - Transformers, Sentence Transformers, and PyTorch:
     ```bash
     pip install transformers sentence-transformers
     ```

2. **Run the Script**:
   ```bash
   python3 zero_shot_classifier_with_similarity.py
   ```

3. **View the Output**:
   - Results are saved as `classification_results_with_similarity.csv` and `classification_results_with_similarity.json` in the script's directory.

Let me know if you need further customization!

The error **"Dimension out of range (expected to be in range of [-1, 0], but got 1)"** typically occurs when the tensor's dimensionality does not match the expected shape in PyTorch operations. This issue arises in the cosine similarity calculation due to inconsistent tensor dimensions, especially when handling variable-length text embeddings from the `sentence-transformers` model.

Let’s address this issue systematically:

---

### **Root Cause**
1. **Empty or Short Text**:
   - The `mean(dim=1)` operation assumes the tensor has at least two dimensions. If the text is empty or very short, the embedding tensor may not meet this requirement.

2. **Tokenization Output**:
   - If tokenization produces a tensor with dimensions that don't match the model's expectation (e.g., a tensor with only one dimension), the `mean(dim=1)` operation will fail.

---

### **Solution**
We can modify the script to:
1. **Validate Text**:
   - Ensure the text is not empty or too short before processing.
2. **Check Embedding Dimensions**:
   - Dynamically handle cases where `mean(dim=1)` fails by adjusting the dimensionality calculation.

---

### **Updated Script Section**

Replace the embedding computation section in the script with this:

```python
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
```

---

### **What Changed**
1. **Text Validation**:
   - Files with empty or very short content are skipped.
   - Added `strip()` to clean unnecessary whitespace.

2. **Embedding Computation**:
   - The `get_text_embedding` function checks tensor dimensionality before applying operations like `mean(dim=1)`.
   - Handles edge cases where tensors have only one dimension.

3. **Error Handling**:
   - Improved error messages for debugging.
   - Skips problematic files without crashing the entire process.

---

### **Next Steps**
1. **Run the Script**:
   Test the updated script on your text files.
2. **Inspect Skipped Files**:
   If files are being skipped frequently, check their contents to ensure they are valid and meaningful.
3. **Debug Output**:
   Add debug print statements to inspect tensor shapes during processing if errors persist.

No problem! Here's the **complete script** with the modification integrated properly. It includes the updated embedding computation logic and handles edge cases to avoid the dimension error. The entire flow is preserved, with added robustness.

---

### **Complete Script**

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

    # Save results to CSV and JSON
    output_csv = "classification_results_with_similarity.csv"
    output_json = "classification_results_with_similarity.json"
    save_results_csv(results, output_csv)
    save_results_json(results, output_json)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")

if __name__ == "__main__":
    main()
```

---

### **What’s New**
1. **Updated `get_text_embedding` Function**:
   - Handles tensors with varying dimensions.
   - Skips invalid or empty text gracefully.

2. **Integrated Validation**:
   - Skips empty or invalid files early to prevent errors.

3. **Cosine Similarity Calculation**:
   - Computes cosine similarity between the document embedding and each label embedding.
   - Includes this score in both the console output and the saved results (`CSV` and `JSON`).

---

### **How to Use**
1. Save this script to a file, e.g., `zero_shot_classifier_with_similarity.py`.
2. Run the script:
   ```bash
   python3 zero_shot_classifier_with_similarity.py
   ```
3. Follow the prompts to:
   - Specify the directory containing `.txt` files.
   - Enter target labels (e.g., `science, law, biology`).
   - Choose a GPU or CPU.

4. Review the output:
   - **Console**: Displays classification scores and cosine similarities.
   - **Files**:
     - `classification_results_with_similarity.csv`: Tabular results.
     - `classification_results_with_similarity.json`: Hierarchical results.

Let me know if you encounter any further issues or need enhancements! 😊
