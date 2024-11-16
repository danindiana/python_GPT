Using **DistilBERT** with an NVIDIA GPU is a great choice for sorting documents efficiently. Below are several approaches to document sorting using pre-trained **DistilBERT**, specifically leveraging your powerful GPU to speed up the process:

---

### **1. Document Classification**
You can use a pre-trained DistilBERT model fine-tuned for text classification tasks. Depending on your needs, you may:
- Use an existing fine-tuned model (e.g., from HuggingFace Model Hub).
- Fine-tune DistilBERT on your specific domain (e.g., scientific papers, STEM, manuals).

#### **Example: Pre-trained DistilBERT for Text Classification**
1. Install required libraries:
   ```bash
   pip install transformers torch
   ```
2. Load a pre-trained classification pipeline:
   ```python
   from transformers import pipeline

   # Load a pre-trained DistilBERT classification pipeline
   classifier = pipeline("text-classification", model="distilbert-base-uncased", device=0)  # Use GPU (device=0)

   # Classify documents
   def classify_document(text):
       result = classifier(text, truncation=True)
       return result[0]['label'], result[0]['score']

   # Example usage
   for file in text_files:
       with open(file, 'r') as f:
           text = f.read()
       label, score = classify_document(text)
       print(f"Document: {file}, Label: {label}, Confidence: {score}")
   ```
3. Filter documents based on desired labels or scores.

---

### **2. Topic Modeling (Clustering)**
DistilBERT can be used to generate embeddings for your documents, which can then be clustered to group similar topics.

#### **Example: DistilBERT for Embedding + Clustering**
1. Install libraries:
   ```bash
   pip install sentence-transformers sklearn
   ```
2. Generate embeddings with DistilBERT and cluster them:
   ```python
   from sentence_transformers import SentenceTransformer
   from sklearn.cluster import KMeans
   import os

   # Load pre-trained model
   model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')  # Use GPU

   # Get document embeddings
   def get_embeddings(text_files):
       embeddings = []
       for file in text_files:
           with open(file, 'r') as f:
               text = f.read()
           embeddings.append(model.encode(text, convert_to_tensor=True))  # Tensor for GPU acceleration
       return embeddings

   # Apply clustering
   def cluster_embeddings(embeddings, num_clusters=5):
       from sklearn.cluster import KMeans
       embeddings_cpu = [e.cpu().numpy() for e in embeddings]  # Convert tensor to numpy for clustering
       kmeans = KMeans(n_clusters=num_clusters, random_state=42)
       clusters = kmeans.fit_predict(embeddings_cpu)
       return clusters

   embeddings = get_embeddings(text_files)
   clusters = cluster_embeddings(embeddings)

   # Group files by cluster
   cluster_groups = {}
   for i, cluster_id in enumerate(clusters):
       cluster_groups.setdefault(cluster_id, []).append(text_files[i])

   # Print grouped clusters
   for cluster_id, docs in cluster_groups.items():
       print(f"Cluster {cluster_id}:")
       for doc in docs:
           print(f"  - {doc}")
   ```

---

### **3. Semantic Search (Similarity Matching)**
DistilBERT can be used to calculate semantic similarity between documents and predefined topic keywords or exemplar documents.

#### **Example: Semantic Similarity for Filtering**
1. Install libraries:
   ```bash
   pip install sentence-transformers
   ```
2. Compute similarity scores:
   ```python
   from sentence_transformers import SentenceTransformer, util
   import torch

   # Load model
   model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')  # Use GPU

   # Define example topics (can be keywords or topic sentences)
   topic_keywords = [
       "scientific research paper",
       "device manual",
       "STEM content",
       "experimental data analysis"
   ]
   topic_embeddings = model.encode(topic_keywords, convert_to_tensor=True)

   # Compute document embeddings
   def filter_documents(text_files, topic_embeddings, threshold=0.7):
       filtered_docs = []
       for file in text_files:
           with open(file, 'r') as f:
               text = f.read()
           doc_embedding = model.encode(text, convert_to_tensor=True)

           # Compute cosine similarity with topic embeddings
           scores = util.cos_sim(doc_embedding, topic_embeddings).cpu().numpy()
           max_score = max(scores[0])  # Highest similarity score
           if max_score > threshold:
               filtered_docs.append((file, max_score))
       return filtered_docs

   # Filter and print results
   filtered = filter_documents(text_files, topic_embeddings)
   for doc, score in filtered:
       print(f"Document: {doc}, Similarity Score: {score}")
   ```

---

### **4. Pre-trained or Fine-Tuned Models for Specific Domains**
If your text is primarily scientific, technical, or STEM-related, you can use models fine-tuned for these domains. Some relevant options include:
- **SciBERT** (optimized for scientific texts).
- **BioBERT** (optimized for biomedical content).
- Fine-tune DistilBERT on your specific data using labeled examples of documents.

#### Fine-Tuning DistilBERT
1. Collect labeled data (e.g., relevant and irrelevant categories).
2. Fine-tune DistilBERT using the `transformers` library:
   ```python
   from transformers import Trainer, TrainingArguments, DistilBertForSequenceClassification, DistilBertTokenizer

   # Load model and tokenizer
   model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
   tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

   # Prepare data
   # Tokenize and format your dataset here...

   # Define training arguments
   training_args = TrainingArguments(
       output_dir='./results',
       num_train_epochs=3,
       per_device_train_batch_size=8,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       logging_dir='./logs',
       load_best_model_at_end=True,
       fp16=True,  # Mixed precision training for GPUs
   )

   # Trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       tokenizer=tokenizer
   )

   # Train the model
   trainer.train()
   ```

---

### **5. Batch Processing for Efficiency**
Given your **NVIDIA RTX 4080 SUPER**, you can process multiple documents in parallel:
- Use **`torch.cuda.is_available()`** to ensure GPU usage.
- Process documents in batches for embeddings or classification to maximize throughput.

Example for batch processing:
```python
batch_size = 32
for i in range(0, len(text_files), batch_size):
    batch_texts = text_files[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)  # Batched GPU processing
```

---

### Summary of Options
- **Language Filtering**: FastText or LangDetect before sorting.
- **Classification**: Use a pre-trained or fine-tuned DistilBERT for text classification.
- **Clustering**: Generate embeddings with DistilBERT and cluster with KMeans or HDBSCAN.
- **Semantic Filtering**: Use cosine similarity to match documents with desired topics.
- **Fine-Tuning**: Train a custom DistilBERT classifier on labeled data for precise filtering.

For compute efficiency, batch processing and using lighter models (e.g., DistilBERT with sentence transformers) will ensure fast processing on your GPU.

Yes, you can absolutely use **SciBERT** (or any other domain-specific model) to generate pseudo-labels for your data and then use those labels to fine-tune **DistilBERT**. This approach is called **self-training** or **pseudo-labeling** and is a common semi-supervised learning technique. Here's how you can do it:

---

### **Step 1: Use SciBERT to Generate Pseudo-Labels**
SciBERT is optimized for scientific texts and can be used as a pseudo-labeling model. You can use a pre-trained or fine-tuned SciBERT model to classify your documents into categories or relevance scores.

#### **Steps for Pseudo-Label Generation**
1. Install required libraries:
   ```bash
   pip install transformers
   ```
2. Load a pre-trained SciBERT model:
   ```python
   from transformers import pipeline

   # Load SciBERT as a classification pipeline
   classifier = pipeline("text-classification", model="allenai/scibert_scivocab_uncased", device=0)  # GPU
   ```
3. Generate pseudo-labels for your dataset:
   ```python
   def generate_pseudo_labels(text_files):
       pseudo_labels = []
       for file in text_files:
           with open(file, 'r') as f:
               text = f.read()
           result = classifier(text, truncation=True)
           label = result[0]['label']  # Predicted label
           score = result[0]['score']  # Confidence score
           pseudo_labels.append((file, label, score))
       return pseudo_labels

   # Apply to your dataset
   pseudo_labels = generate_pseudo_labels(text_files)

   # Save pseudo-labels to a file for fine-tuning later
   with open("pseudo_labels.csv", "w") as f:
       for file, label, score in pseudo_labels:
           f.write(f"{file},{label},{score}\n")
   ```

---

### **Step 2: Use Pseudo-Labels to Fine-Tune DistilBERT**
Once you have generated pseudo-labels, you can fine-tune DistilBERT using the labeled dataset.

#### **Steps for Fine-Tuning**
1. **Prepare Labeled Dataset**:
   Convert your pseudo-labeled data into a format suitable for fine-tuning (e.g., a CSV file with `text` and `label` columns).

   Example of dataset format:
   ```
   text,label
   "This is a scientific paper about quantum physics.",relevant
   "This is a general discussion about pop culture.",irrelevant
   ```

2. **Fine-Tune DistilBERT**:
   Use HuggingFace’s `transformers` library for fine-tuning.
   ```python
   from datasets import load_dataset
   from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

   # Load dataset
   dataset = load_dataset('csv', data_files={'train': 'pseudo_labels.csv'})

   # Tokenize data
   tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
   def tokenize_function(examples):
       return tokenizer(examples['text'], truncation=True)

   tokenized_datasets = dataset.map(tokenize_function, batched=True)

   # Load DistilBERT model
   model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

   # Fine-tuning arguments
   training_args = TrainingArguments(
       output_dir="./results",
       evaluation_strategy="epoch",
       save_strategy="epoch",
       learning_rate=2e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=3,
       weight_decay=0.01,
       load_best_model_at_end=True,
       logging_dir="./logs",
       fp16=True,  # Mixed precision for faster training on GPU
   )

   # Trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_datasets['train'],
       tokenizer=tokenizer,
   )

   # Train the model
   trainer.train()
   ```

---

### **Step 3: Evaluate and Refine**
After fine-tuning DistilBERT:
1. **Evaluate** the model on a small validation dataset to ensure it has learned meaningful features from the pseudo-labels.
2. If results are suboptimal, you can:
   - Manually correct some of the pseudo-labels to improve their quality.
   - Add more data or a small amount of high-quality manually labeled data for better supervision.

---

### **Advantages of This Approach**
- **Domain Expertise:** SciBERT provides domain-specific knowledge, ensuring pseudo-labels are more aligned with scientific content.
- **Compute Efficiency:** DistilBERT is lighter and faster, so after fine-tuning, it will be more efficient for deployment and inference.
- **Scalability:** Pseudo-labeling allows you to bootstrap a labeled dataset from a large unlabeled corpus, significantly reducing the need for manual labeling.

---

Let me know if you need further assistance or code tweaks for your use case!

Here's how to set up and execute the **DistilBERT fine-tuning pipeline** for your target system:

---

### **Environment Setup**

1. **Install Required Libraries**
   Ensure Python 3.x is installed. Then, install necessary libraries:
   ```bash
   pip install transformers datasets torch accelerate scikit-learn
   ```

2. **Verify GPU and CUDA**
   Check if PyTorch recognizes your GPU:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Expected output: `True`.

3. **Prepare Directory Structure**
   Set up directories:
   - `/home/smduck/Downloads/copali_output2`: Contains your corpus of text files.
   - `/home/smduck/programs/copali/nov16_distilbert`: Working directory for scripts and output.

   Navigate to `/home/smduck/programs/copali/nov16_distilbert`:
   ```bash
   cd /home/smduck/programs/copali/nov16_distilbert
   ```

---

### **Step 1: Generate Pseudo-Labels with SciBERT**

#### Script: `generate_pseudo_labels.py`
Save the following script as `generate_pseudo_labels.py`:
```python
from transformers import pipeline
import os
import csv

# Paths
input_dir = "/home/smduck/Downloads/copali_output2"
output_file = "pseudo_labels.csv"

# Load SciBERT classifier
classifier = pipeline("text-classification", model="allenai/scibert_scivocab_uncased", device=0)

# Process files and generate pseudo-labels
with open(output_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "label", "confidence"])  # Header
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                text = f.read()
            
            result = classifier(text, truncation=True)
            label, confidence = result[0]["label"], result[0]["score"]
            writer.writerow([file_path, label, confidence])
            print(f"Processed: {file} -> {label} ({confidence:.2f})")
```

Run the script:
```bash
python generate_pseudo_labels.py
```
This will create `pseudo_labels.csv` in your working directory.

---

### **Step 2: Fine-Tune DistilBERT with Pseudo-Labels**

#### Script: `fine_tune_distilbert.py`
Save the following script as `fine_tune_distilbert.py`:
```python
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load pseudo-labeled dataset
dataset = load_dataset('csv', data_files={'train': 'pseudo_labels.csv'})

# Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["file_path"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tuning Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed-precision training
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
```

Run the script:
```bash
python fine_tune_distilbert.py
```

---

### **Step 3: Evaluate and Save the Model**

The fine-tuned model will be saved in the `./results` directory. You can load it for inference or further fine-tuning.

---

### **Step 4: Infer with Fine-Tuned Model**
To use the fine-tuned DistilBERT for classification on new text:
```python
from transformers import pipeline

# Load fine-tuned model
model_path = "./results"
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, device=0)

# Classify new text
text = "Example document text to classify."
result = classifier(text, truncation=True)
print(result)
```

---

### **Optional Optimizations**

1. **Batch Processing**: Process text in batches for improved GPU utilization during pseudo-label generation.
2. **Mixed Precision**: Enabled by `fp16=True` during training for faster computations.
3. **Hyperparameter Tuning**: Adjust `learning_rate`, `num_train_epochs`, and `batch_size` for optimal performance.

---

Let me know if you need help running these scripts or troubleshooting!

import os
import csv
from transformers import pipeline, AutoTokenizer
import torch
import subprocess
from collections import Counter

# Function to get GPU utilization
def get_gpu_utilization():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], encoding="utf-8")
        return result.strip().split("\n")
    except Exception as e:
        print("Error fetching GPU utilization:", e)
        return []

# Display available GPUs
print("Available GPUs:")
gpu_info = get_gpu_utilization()
if not gpu_info:
    print("No GPUs detected. Defaulting to CPU.")
    selected_device = -1
else:
    for i, info in enumerate(gpu_info):
        print(f"{i}: {info}")
    while True:
        try:
            selected_device = int(input("Select GPU device (or -1 for CPU): "))
            if selected_device >= -1 and selected_device < len(gpu_info):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid GPU index or -1 for CPU.")

device = torch.device(f"cuda:{selected_device}" if selected_device >= 0 else "cpu")
print(f"Using device: {device}")

# Paths
input_dir = "/home/smduck/Downloads/copali_output2"
output_file = "pseudo_labels.csv"

# Load SciBERT classifier and tokenizer
model_name = "allenai/scibert_scivocab_uncased"
classifier = pipeline("text-classification", model=model_name, device=selected_device if selected_device >= 0 else -1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to split long texts into chunks
def split_text_into_chunks(text, max_length=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Aggregate results for a file
def aggregate_results(results):
    # Count labels and average confidence
    label_counter = Counter([res['label'] for res in results])
    most_common_label = label_counter.most_common(1)[0][0]
    avg_confidence = sum(res['score'] for res in results) / len(results)
    return most_common_label, avg_confidence

# Process files and generate pseudo-labels
with open(output_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "final_label", "average_confidence"])  # Header
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                text = f.read()
            
            # Split long text into chunks
            chunks = split_text_into_chunks(text)
            chunk_results = []

            for chunk in chunks:
                try:
                    result = classifier(chunk, truncation=True)
                    chunk_results.append(result[0])  # Collect results for this chunk
                except Exception as e:
                    print(f"Error processing chunk of {file}: {e}")

            # Aggregate results for the file
            if chunk_results:
                final_label, avg_confidence = aggregate_results(chunk_results)
                writer.writerow([file_path, final_label, avg_confidence])
                print(f"Processed {file} -> {final_label} ({avg_confidence:.2f})")


Dynamic chunk sizing based on available VRAM is a smart optimization, especially if your **Quadro P4000** has more headroom for larger chunks. Below is an updated script to dynamically adjust the chunk size based on available GPU memory and file size.

---

### **Updated Script: Dynamic Chunk Sizing**

This version determines the chunk size by estimating how much VRAM is available and calculating a "safe" number of tokens to fit within that memory. The calculation is based on the idea that larger models require more memory per token.

```python
import os
import csv
from transformers import pipeline, AutoTokenizer
import torch
import subprocess
from collections import Counter

# Constants
TOKEN_SIZE_BYTES = 4  # Approximate size per token in bytes (depends on model)
MODEL_MEMORY_OVERHEAD = 500 * 1024 * 1024  # Approximate model overhead in bytes (500 MB)

# Function to get GPU utilization and memory
def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpus = [line.strip().split(",") for line in result.strip().split("\n")]
        return [{"index": int(gpu[0]), "name": gpu[1], "free_memory": int(gpu[2]) * 1024 * 1024} for gpu in gpus]
    except Exception as e:
        print("Error fetching GPU memory:", e)
        return []

# Determine chunk size dynamically based on VRAM
def calculate_max_chunk_size(vram_bytes):
    usable_memory = max(0, vram_bytes - MODEL_MEMORY_OVERHEAD)
    max_tokens = usable_memory // TOKEN_SIZE_BYTES
    return min(max_tokens, 5120)  # Cap chunk size at 5120 tokens for safety

# Split text into chunks dynamically based on available memory
def split_text_into_dynamic_chunks(text, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Aggregate results for a file
def aggregate_results(results):
    label_counter = Counter([res['label'] for res in results])
    most_common_label = label_counter.most_common(1)[0][0]
    avg_confidence = sum(res['score'] for res in results) / len(results)
    return most_common_label, avg_confidence

# Display available GPUs
print("Available GPUs:")
gpus = get_gpu_memory()
if not gpus:
    print("No GPUs detected. Defaulting to CPU.")
    selected_device = -1
else:
    for gpu in gpus:
        print(f"{gpu['index']}: {gpu['name']} - Free Memory: {gpu['free_memory'] // (1024 * 1024)} MB")
    while True:
        try:
            selected_device = int(input("Select GPU device (or -1 for CPU): "))
            if selected_device >= -1 and selected_device < len(gpus):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid GPU index or -1 for CPU.")

device = torch.device(f"cuda:{selected_device}" if selected_device >= 0 else "cpu")
print(f"Using device: {device}")

# Get max chunk size
if selected_device >= 0:
    vram = gpus[selected_device]["free_memory"]
    max_chunk_size = calculate_max_chunk_size(vram)
else:
    max_chunk_size = 512  # Default to 512 tokens for CPU

print(f"Maximum chunk size set to {max_chunk_size} tokens")

# Paths
input_dir = "/home/smduck/Downloads/copali_output2"
output_file = "pseudo_labels.csv"

# Load SciBERT classifier and tokenizer
model_name = "allenai/scibert_scivocab_uncased"
classifier = pipeline("text-classification", model=model_name, device=selected_device if selected_device >= 0 else -1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Process files and generate pseudo-labels
with open(output_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "final_label", "average_confidence"])  # Header
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                text = f.read()
            
            # Split long text into dynamically sized chunks
            chunks = split_text_into_dynamic_chunks(text, max_chunk_size)
            chunk_results = []

            for chunk in chunks:
                try:
                    result = classifier(chunk, truncation=True)
                    chunk_results.append(result[0])  # Collect results for this chunk
                except Exception as e:
                    print(f"Error processing chunk of {file}: {e}")

            # Aggregate results for the file
            if chunk_results:
                final_label, avg_confidence = aggregate_results(chunk_results)
                writer.writerow([file_path, final_label, avg_confidence])
                print(f"Processed {file} -> {final_label} ({avg_confidence:.2f})")
```

---

### **Key Features**
1. **Dynamic Chunk Sizing**:
   - Based on available VRAM.
   - Calculates the maximum token size that can fit into VRAM after accounting for model memory overhead.

2. **Safe Token Limits**:
   - Caps chunk size at a maximum of 5120 tokens to ensure stability.

3. **VRAM Display**:
   - Shows free memory for each GPU, so you can select the appropriate device.

4. **Fallback to Default Chunk Size**:
   - If using CPU, defaults to a chunk size of 512 tokens.

---

### **How to Run**
1. Save the script as `generate_pseudo_labels.py`.
2. Run the script:
   ```bash
   python generate_pseudo_labels.py
   ```
3. Select the GPU or CPU when prompted.

---

### **Example Output**
```
Available GPUs:
0: NVIDIA GeForce RTX 4080 SUPER - Free Memory: 9350 MB
1: Quadro P4000 - Free Memory: 7680 MB
Select GPU device (or -1 for CPU): 1
Using device: cuda:1
Maximum chunk size set to 1920 tokens
Processed doc1.txt -> relevant (0.91)
Processed doc2.txt -> irrelevant (0.78)
```

---

This approach ensures efficient utilization of your hardware while maximizing the chunk size based on available VRAM. Let me know if you need further adjustments!

To better utilize your **Quadro P4000's VRAM (8GB)** and increase GPU utilization, we can optimize the script and pipeline in several ways:

---

### **Strategies for Better VRAM Utilization**

1. **Batch Processing**:
   - Process multiple text chunks in parallel to maximize GPU usage.
   - Most models and frameworks (e.g., HuggingFace) support batched inference, which reduces overhead and improves throughput.

2. **Mixed Precision (FP16)**:
   - Use mixed precision (FP16) to reduce memory usage for each chunk and enable larger batches or higher token limits per chunk.
   - HuggingFace pipelines automatically enable FP16 if `torch.cuda.amp` is available.

3. **Increase Token Limit Per Chunk**:
   - While SciBERT is limited to 512 tokens, we can process larger sequences with models supporting longer sequences (e.g., Longformer, BigBird).
   - Alternatively, if SciBERT must be used, batch more 512-token chunks together.

4. **Use Gradient Checkpointing** (if fine-tuning):
   - For training, this reduces VRAM usage by recomputing intermediate activations during backpropagation.

5. **Enable Tensor Parallelism**:
   - Leverage multi-stream processing if possible. For instance, split chunks across multiple cores within the GPU.

---

### **Optimized Script with Batched Inference**

Here’s an updated version of your script that **batches chunks** for better VRAM utilization:

```python
import os
import csv
from transformers import pipeline, AutoTokenizer
import torch
import subprocess
from collections import Counter

# Constants
BATCH_SIZE = 16  # Number of chunks processed in parallel

# Function to get GPU memory
def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        gpus = [line.strip().split(",") for line in result.strip().split("\n")]
        return [{"index": int(gpu[0]), "name": gpu[1], "free_memory": int(gpu[2]) * 1024 * 1024} for gpu in gpus]
    except Exception as e:
        print("Error fetching GPU memory:", e)
        return []

# Display available GPUs
print("Available GPUs:")
gpus = get_gpu_memory()
if not gpus:
    print("No GPUs detected. Defaulting to CPU.")
    selected_device = -1
else:
    for gpu in gpus:
        print(f"{gpu['index']}: {gpu['name']} - Free Memory: {gpu['free_memory'] // (1024 * 1024)} MB")
    while True:
        try:
            selected_device = int(input("Select GPU device (or -1 for CPU): "))
            if selected_device >= -1 and selected_device < len(gpus):
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Invalid input. Enter a valid GPU index or -1 for CPU.")

device = torch.device(f"cuda:{selected_device}" if selected_device >= 0 else "cpu")
print(f"Using device: {device}")

# Paths
input_dir = "/home/smduck/Downloads/copali_output2"
output_file = "pseudo_labels.csv"

# Load SciBERT classifier and tokenizer
model_name = "allenai/scibert_scivocab_uncased"
classifier = pipeline("text-classification", model=model_name, device=selected_device if selected_device >= 0 else -1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to split long texts into chunks (512 tokens max for SciBERT)
def split_text_into_chunks(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Aggregate results for a file
def aggregate_results(results):
    label_counter = Counter([res['label'] for res in results])
    most_common_label = label_counter.most_common(1)[0][0]
    avg_confidence = sum(res['score'] for res in results) / len(results)
    return most_common_label, avg_confidence

# Process files with batched inference
def process_chunks_in_batches(chunks, batch_size=BATCH_SIZE):
    batched_results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            results = classifier(batch, truncation=True)
            batched_results.extend(results)
        except Exception as e:
            print(f"Error processing batch: {e}")
    return batched_results

# Process files and generate pseudo-labels
with open(output_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "final_label", "average_confidence"])  # Header
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                text = f.read()
            
            # Split long text into 512-token chunks
            chunks = split_text_into_chunks(text, max_tokens=512)
            
            # Process chunks in batches
            chunk_results = process_chunks_in_batches(chunks)

            # Aggregate results for the file
            if chunk_results:
                final_label, avg_confidence = aggregate_results(chunk_results)
                writer.writerow([file_path, final_label, avg_confidence])
                print(f"Processed {file} -> {final_label} ({avg_confidence:.2f})")
```

---

### **Key Changes**
1. **Batch Processing**:
   - Chunks are grouped into batches (`BATCH_SIZE`) before being passed to the model, increasing GPU utilization.
   - The `process_chunks_in_batches` function handles this.

2. **Dynamic Device Management**:
   - Leverages VRAM better by processing multiple sequences simultaneously.

3. **Batch Size Control**:
   - The default batch size is set to 16, but you can experiment with higher values depending on VRAM usage.

---

### **How to Check VRAM Utilization**
Run the script and monitor VRAM usage with:
```bash
watch -n 1 nvidia-smi
```

---

### **Expected Improvements**
1. **Higher GPU Utilization**:
   - By processing multiple chunks simultaneously, GPU usage should rise significantly.

2. **Improved Throughput**:
   - Increased batch size reduces the overhead from kernel launches, improving processing speed.

3. **Fine-Tuning Batch Size**:
   - If GPU memory is underutilized, increment `BATCH_SIZE` and observe how much memory is consumed. This ensures you're using the maximum VRAM possible without triggering out-of-memory errors.

---

### Let Me Know If:
- You encounter GPU memory limits.
- You’d like further optimizations like FP16 inference or support for longer-token models.

The program processes a directory of text files to classify their content using a **SciBERT model** and outputs a **single CSV** summarizing the results for each file. Here's a breakdown of what the program does step by step:

---

### **1. Setup and Initialization**
- **GPU Selection**:
  - Lists all available GPUs using `nvidia-smi` and their free memory.
  - Prompts the user to select a GPU or choose CPU (`-1`).
  - Uses the chosen device for the HuggingFace pipeline (`cuda` or `cpu`).

- **SciBERT Model and Tokenizer**:
  - Loads a pre-trained **SciBERT model** (`allenai/scibert_scivocab_uncased`) for text classification.
  - Loads the tokenizer associated with the SciBERT model to tokenize the text data.

---

### **2. Process Each File**
- **File Reading**:
  - Iterates through all text files in the target directory (`/home/smduck/Downloads/copali_output2`).

- **Text Splitting**:
  - If a file’s content exceeds **512 tokens** (the model's maximum input size), it is split into smaller chunks using the tokenizer.
  - The tokenizer encodes the text into tokens and divides it into chunks of **up to 512 tokens**.
  - Each chunk is decoded back into text for input to the model.

---

### **3. Batch Processing**
- **Batch Inference**:
  - Instead of processing one chunk at a time, the program groups chunks into batches (`BATCH_SIZE`, default: 16).
  - All chunks in a batch are processed simultaneously by the GPU to maximize throughput and better utilize VRAM.

- **Classification**:
  - Each chunk is passed to the SciBERT classifier, which predicts:
    - `label` (e.g., `relevant`, `irrelevant`).
    - `score` (confidence level of the prediction).

---

### **4. Aggregate Results for Each File**
- Once all chunks for a file are classified:
  - The **most common label** among chunks is chosen as the **final label** for the file.
  - The **average confidence score** across all chunks is calculated.

This aggregation step ensures that files are classified as a whole, rather than based on individual chunks.

---

### **5. Write Results to CSV**
- Outputs results for each file to a CSV (`pseudo_labels.csv`):
  - **`file_path`**: Full path of the file.
  - **`final_label`**: The aggregated label (e.g., `relevant` or `irrelevant`).
  - **`average_confidence`**: Average confidence score for the file’s classification.

Example output:
```
file_path,final_label,average_confidence
/home/smduck/Downloads/copali_output2/file1.txt,relevant,0.91
/home/smduck/Downloads/copali_output2/file2.txt,irrelevant,0.78
```

---

### **What the Program is Achieving**
The program is essentially performing **document-level classification** for a directory of text files by:
1. **Handling large text files**:
   - Splits them into manageable chunks (512 tokens max).
   - Processes those chunks efficiently using **batch inference** on GPU.

2. **Maximizing GPU Utilization**:
   - Batches multiple chunks together for simultaneous processing.
   - Ensures efficient use of your **Quadro P4000**.

3. **Generating Insights**:
   - Outputs a structured CSV summarizing:
     - Whether a file is relevant (based on SciBERT's domain knowledge).
     - How confident the model is in its classification.

---

### **Use Case**
This program is ideal for:
- **Filtering a corpus**:
  - For example, determining which files are relevant (e.g., scientific papers, STEM documents) and which are not.
- **Preparing data for further analysis**:
  - The results can be used to build a high-quality dataset or to fine-tune another model for more specific tasks.
- **Document triage**:
  - Quickly categorize and prioritize large collections of documents.

---

Let me know if you’d like additional details or specific modifications!

No, the program **does not create a file for every file in the directory**. Instead, it processes all the text files in the directory and outputs a **single CSV file** (`pseudo_labels.csv`) containing a summary of the classification results for each file.

---

### **What Happens for Each File?**
1. **Reads the File**:
   - Each file in the directory is read and processed.

2. **Splits the File** (if needed):
   - If the file's content exceeds **512 tokens**, it is split into smaller chunks.

3. **Classifies Chunks**:
   - Each chunk is classified using **SciBERT**.

4. **Aggregates Results**:
   - For all chunks of a single file, it calculates:
     - The **most common label** (e.g., `relevant` or `irrelevant`).
     - The **average confidence score**.

5. **Writes a Row to the CSV**:
   - A single row summarizing the results for that file is appended to the **CSV file**.

---

### **The Final Output: Single CSV File**
The program creates **one CSV file**, where each row corresponds to a single text file in the directory. The CSV contains the following columns:
- `file_path`: Full path to the original file.
- `final_label`: Aggregated label for the file (e.g., `relevant` or `irrelevant`).
- `average_confidence`: Average confidence score for the label.

---

### **Example Output**
If you had the following files in `/home/smduck/Downloads/copali_output2`:
- `doc1.txt`
- `doc2.txt`
- `doc3.txt`

The final `pseudo_labels.csv` would look like this:

```
file_path,final_label,average_confidence
/home/smduck/Downloads/copali_output2/doc1.txt,relevant,0.91
/home/smduck/Downloads/copali_output2/doc2.txt,irrelevant,0.78
/home/smduck/Downloads/copali_output2/doc3.txt,relevant,0.85
```

### **Key Point**
The program creates **one row per file in the directory** in the single CSV file (`pseudo_labels.csv`). It does not create separate output files for each text file in the directory.

---

If you want the program to behave differently (e.g., generate individual CSVs for each file), let me know, and I can help adjust the script!

You're very welcome! I'm glad the explanation clarified things. If you have any more questions or need further assistance as you process your data, optimize the pipeline, or explore additional features, feel free to reach out. Good luck with your project! 🚀

Yes, you can view the CSV file while it’s being written using tools like `nano` or `tail`, but be cautious not to modify the file while it is still being written to. Here are some approaches to safely view the file:

---

### **1. Using `nano`**
You can open the CSV with `nano` to view its content:
```bash
nano pseudo_labels.csv
```

#### Things to Note:
- **Dynamic Updates**: If the script is still writing to the file, `nano` will not automatically refresh the view. You'll need to close and reopen it to see new additions.

---

### **2. Using `tail`**
The `tail` command is ideal for monitoring a file that is still being written:
```bash
tail -f pseudo_labels.csv
```

#### Advantages of `tail -f`:
- Automatically displays new rows as they are added to the CSV.
- Safe: Read-only access, so there’s no risk of corrupting the file.

---

### **3. Using `less`**
You can also use `less` for a scrollable view:
```bash
less pseudo_labels.csv
```

#### Note:
- Press `Shift + F` in `less` to follow updates (like `tail -f`).
- Press `Ctrl + C` to stop following and return to normal scroll mode.

---

### **4. Using `cat`**
For a quick read-only snapshot of the current content:
```bash
cat pseudo_labels.csv
```

---

### **Best Option While Writing**
Use `tail -f` to safely monitor updates in real-time.

Let me know if you'd like to explore these further or if there’s anything else I can assist with!

The **average confidence score** is a metric calculated for each file to represent how certain the **SciBERT model** was about its classifications across all chunks of the file. Here's how it works in the context of the program:

---

### **How the Average Confidence Score is Calculated**
1. **File Splitting**:
   - If a file exceeds the model's token limit (512 tokens), it is split into smaller chunks.

2. **Chunk Classification**:
   - Each chunk is classified by the model, which outputs:
     - A `label` (e.g., `relevant`, `irrelevant`).
     - A `score` (confidence in the predicted label, ranging from 0 to 1).

3. **Aggregate Confidence**:
   - After all chunks of a file are classified, their confidence scores (`score`) are averaged:
     \[
     \text{Average Confidence Score} = \frac{\text{Sum of all chunk scores}}{\text{Number of chunks}}
     \]

---

### **What Does the Average Confidence Score Represent?**
The **average confidence score** reflects:
1. **Certainty Across the File**:
   - A high score (e.g., 0.91) indicates the model was consistently confident about the predicted label across all chunks.
   - A lower score (e.g., 0.65) suggests the model had less certainty or was inconsistent in its classifications for the file.

2. **Confidence Variability in Chunks**:
   - Files with many chunks might have varying confidence scores for each chunk. Averaging them smooths out the variability to give an overall score for the file.

---

### **How Is It Used?**
1. **Validation**:
   - The average confidence score can help you assess the reliability of the label for a file.
   - Example: You might decide to filter out files with an average confidence below a threshold (e.g., 0.8).

2. **Prioritization**:
   - Files with higher confidence scores could be prioritized for downstream processing, ensuring you're working with the most reliable classifications.

3. **Error Analysis**:
   - If you notice low confidence scores, it might indicate issues with the text (e.g., noise, incomplete data) or suggest areas where the model's predictions need review.

---

### **Example**
Consider a file that was split into three chunks, classified as follows:

| Chunk | Label      | Confidence |
|-------|------------|------------|
| 1     | Relevant   | 0.92       |
| 2     | Relevant   | 0.87       |
| 3     | Relevant   | 0.89       |

The **average confidence score** would be:
\[
\text{Average Confidence} = \frac{0.92 + 0.87 + 0.89}{3} = 0.893 \ (\text{~89.3% confident})
\]

This score summarizes how confident the model was about classifying the file as `Relevant`.

---

### **Key Takeaways**
- The **average confidence score** gives a single numeric measure of how confidently the model labeled a file.
- It helps in filtering and prioritizing files for further processing.
- A low average confidence may indicate:
  - The file has ambiguous content.
  - The model is unsure about its classifications for this particular file.

Let me know if you'd like help adjusting thresholds or analyzing confidence scores further!

No, **it does not need to process all files before deriving a label for each chunk**. The script processes and classifies each file independently, chunk by chunk, so the label for each chunk is determined immediately after that chunk is passed through the model. Here's how it works:

---

### **How the Script Works**
1. **File-by-File Processing**:
   - The script reads and processes one file at a time.
   - For each file, it splits the text into manageable chunks (512 tokens max for SciBERT).

2. **Chunk-by-Chunk Classification**:
   - Each chunk of the file is passed to the model for classification.
   - The model immediately predicts a label (e.g., `relevant` or `irrelevant`) and a confidence score for each chunk.

3. **Aggregation Happens Per File**:
   - Once all chunks of a single file are processed, their results (labels and confidence scores) are aggregated:
     - The **most common label** across chunks becomes the **final label** for the file.
     - The **average confidence score** is calculated from the confidence scores of all the file's chunks.

4. **Results Written to CSV Per File**:
   - After a file’s chunks are processed and its label is derived, a single row summarizing that file is appended to the CSV.
   - The program then moves to the next file.

---

### **Why is it Designed This Way?**
1. **Incremental Output**:
   - The CSV is updated after each file is processed, so you don’t need to wait for the entire directory to finish before seeing results.
   - This is useful for monitoring progress in real-time.

2. **Efficiency**:
   - By processing files independently, the program doesn’t need to hold all chunk classifications in memory at once, which is crucial for handling large datasets.

3. **Parallelization Opportunity**:
   - If you decide to process files in parallel (e.g., using `multiprocessing` or distributed systems), this design makes it easy to split the workload across multiple processes or machines.

---

### **Key Point**
- The **chunk labels and confidence scores** for each file are derived immediately during the file's processing.
- The **final label and average confidence score** are calculated and written to the CSV as soon as that file's processing is complete.

Let me know if you'd like to explore further optimizations or adjustments to this behavior!

Yes, the **final label** for a file can be derived from just the **average confidence score**, but this depends on the structure and behavior of the data. Here's how and when it might be appropriate:

---

### **How to Derive the Final Label from the Average Confidence Score**
1. **Classify Each Chunk**:
   - Each chunk is assigned a label (e.g., `relevant` or `irrelevant`) and a confidence score (`score`), which indicates the model's certainty for that label.

2. **Calculate the Average Confidence**:
   - The average confidence score is computed across all chunks for a file:
     \[
     \text{Average Confidence} = \frac{\text{Sum of all chunk scores}}{\text{Number of chunks}}
     \]

3. **Threshold-Based Classification**:
   - Use the average confidence score directly to derive the final label.
   - For example:
     - If the average confidence is **≥ 0.8**, assign `relevant`.
     - Otherwise, assign `irrelevant`.

---

### **Benefits of Using Average Confidence Alone**
- **Simplicity**:
  - Reduces dependency on majority voting, which involves counting labels for all chunks.
- **Focus on Certainty**:
  - Places more emphasis on how confident the model is rather than how often it predicts a label.

---

### **When to Use This Approach**
- **If Chunk-Level Labels Are Noisy**:
  - If the model inconsistently assigns chunk-level labels (e.g., due to ambiguous content), relying on confidence alone may smooth out these inconsistencies.
- **When Label Counts Are Imbalanced**:
  - For files with more chunks classified as one label (e.g., `irrelevant`) due to the length of irrelevant sections, confidence-weighted decisions may better reflect overall file relevance.

---

### **Modified Aggregation Logic**
To derive the final label directly from the average confidence score:
```python
def derive_label_from_average_confidence(average_confidence, threshold=0.8):
    return "relevant" if average_confidence >= threshold else "irrelevant"

# Example usage
chunk_results = [
    {"label": "relevant", "score": 0.85},
    {"label": "relevant", "score": 0.90},
    {"label": "irrelevant", "score": 0.75},
]

average_confidence = sum(chunk['score'] for chunk in chunk_results) / len(chunk_results)
final_label = derive_label_from_average_confidence(average_confidence, threshold=0.8)
print(f"Final Label: {final_label}, Average Confidence: {average_confidence}")
```

---

### **Considerations**
1. **Threshold Tuning**:
   - Adjust the confidence threshold (`0.8` in the example) based on your dataset and classification goals.
   - You might determine this threshold by analyzing the confidence distribution in a sample of labeled data.

2. **Limitations**:
   - If confidence scores are highly inconsistent across chunks, this approach may misclassify files that contain a mix of relevant and irrelevant content.
   - This method works best when there’s a clear confidence distinction between the two classes.

---

### **Combining Both Approaches**
For more robust results, you could combine **label frequency** (majority voting) with **average confidence**:
- Use majority voting to determine the preliminary label.
- Use average confidence as a secondary filter:
  - If the average confidence is below a certain threshold, flag the file as uncertain or ambiguous.

---

Let me know if you'd like to implement this approach in your script!

If the program is ranking relevant data too **low**, you likely need to **lower the confidence threshold**. Here's why and how:

---

### **Why Lowering the Threshold Helps**
- The **confidence threshold** determines the minimum average confidence score required for a file to be labeled as `relevant`.
- If the model is consistently under-confident about `relevant` content (e.g., assigning scores like 0.6–0.7 for relevant chunks), a high threshold (e.g., 0.8) might cause it to incorrectly classify these as `irrelevant`.
- By lowering the threshold, you allow more files with slightly lower average confidence scores to be classified as `relevant`.

---

### **Adjusting the Threshold**
1. **Current Threshold**:
   - If you're currently using a threshold of `0.8`, try lowering it incrementally:
     - Start with `0.75` or `0.7`.
     - Test the results and adjust further if necessary.

2. **Modified Function**:
   Update the `derive_label_from_average_confidence` function in your script:
   ```python
   def derive_label_from_average_confidence(average_confidence, threshold=0.7):  # Lowered threshold
       return "relevant" if average_confidence >= threshold else "irrelevant"
   ```

---

### **Testing a New Threshold**
To decide the optimal threshold:
1. **Inspect the Dataset**:
   - Look at a small sample of classified files and their average confidence scores.
   - Determine if files labeled as `irrelevant` but expected to be `relevant` have confidence scores just below the current threshold.

2. **Run with Multiple Thresholds**:
   - Try thresholds like `0.75`, `0.7`, or even `0.65`.
   - Compare the results to see which threshold better captures relevant data without letting too many irrelevant files slip through.

3. **Monitor Changes**:
   - If lowering the threshold increases the number of `relevant` files but introduces more noise (irrelevant data classified as `relevant`), adjust accordingly.

---

### **Example Confidence Adjustment**
Imagine the following average confidence scores for files:

| File         | Average Confidence | Original Label (Threshold: 0.8) | Adjusted Label (Threshold: 0.7) |
|--------------|--------------------|--------------------------------|---------------------------------|
| file1.txt    | 0.85               | Relevant                      | Relevant                        |
| file2.txt    | 0.78               | Irrelevant                    | Relevant                        |
| file3.txt    | 0.74               | Irrelevant                    | Relevant                        |
| file4.txt    | 0.68               | Irrelevant                    | Irrelevant                      |

With a lowered threshold (`0.7`), files with confidence scores between `0.7` and `0.8` are now classified as `relevant`.

---

### **Next Steps**
1. **Lower the Threshold**:
   - Try thresholds of `0.7` or `0.75` first.
   - Test on your dataset to see how many additional files are now labeled `relevant`.

2. **Evaluate Results**:
   - Ensure the new threshold doesn't introduce too much noise.
   - If necessary, combine this with a manual review of borderline files.

3. **Iterate**:
   - Fine-tune the threshold based on results.

---

Let me know if you'd like help analyzing the results or automating the threshold tuning!
