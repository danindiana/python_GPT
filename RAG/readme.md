A Vector Database-Based Document Search is a powerful approach for Retrieval-Augmented Generation (RAG) pipelines, especially when working with Large Language Models (LLMs). Here are the key benefits:

### 1. **Semantic Search Capabilities**
- **Contextual Understanding**: Traditional keyword-based search methods often fail to capture the nuanced meaning of queries. Vector databases use embeddings (numerical representations of text) to perform semantic search, which means they can understand the context and intent behind queries.
- **Improved Relevance**: By matching the semantic similarity between queries and documents, vector databases can retrieve more relevant documents, even if the exact keywords are not present.

### 2. **Scalability**
- **Efficient Storage and Retrieval**: Vector databases are optimized for storing and querying high-dimensional vectors. They use techniques like indexing and clustering to efficiently manage large datasets, making them scalable for enterprise-level applications.
- **Real-Time Processing**: They can handle real-time queries and updates, which is crucial for dynamic environments where new documents are frequently added.

### 3. **Enhanced LLM Performance**
- **Better Contextual Information**: By providing more relevant and contextual documents, vector databases enhance the performance of LLMs. This ensures that the LLM has access to the most pertinent information, leading to more accurate and contextually appropriate responses.
- **Reduced Hallucination**: LLMs can sometimes generate responses that are not grounded in factual information (hallucination). A vector database can provide reliable, contextually relevant documents to ground the LLM's responses, reducing the likelihood of hallucination.

### 4. **Flexibility and Adaptability**
- **Support for Various Data Types**: Vector databases can handle not just text but also images, audio, and other types of data by converting them into vectors. This makes them versatile for multi-modal applications.
- **Customizable Embeddings**: Users can choose or even train custom embeddings tailored to their specific domain or use case, improving the relevance of search results.

### 5. **Integration with RAG Pipelines**
- **Seamless Integration**: Vector databases integrate smoothly with RAG pipelines, which typically involve embedding generation, document retrieval, and response generation. This integration ensures a cohesive and efficient workflow.
- **Dynamic Retrieval**: The ability to dynamically retrieve and update documents based on user queries allows for a more interactive and responsive user experience.

### 6. **Cost Efficiency**
- **Reduced Computational Overhead**: By efficiently retrieving only the most relevant documents, vector databases reduce the computational overhead associated with processing large volumes of data. This can lead to cost savings in terms of processing power and time.
- **Optimized Resource Utilization**: The efficient storage and retrieval mechanisms of vector databases ensure that resources are used optimally, further contributing to cost efficiency.

### 7. **Improved User Experience**
- **Precise and Relevant Results**: Users receive more precise and relevant search results, enhancing their overall experience. This is particularly important in applications like customer support, where quick and accurate responses are critical.
- **Personalization**: Vector databases can be used to personalize search results based on user preferences and historical interactions, providing a more tailored experience.

### Conclusion
In summary, a Vector Database-Based Document Search offers significant advantages for RAG LLM pipelines, including improved semantic understanding, scalability, enhanced LLM performance, flexibility, seamless integration, cost efficiency, and an improved user experience. These benefits make it a compelling choice for modern AI applications that require robust, scalable, and contextually aware document retrieval.

### Semantic Search Capabilities and Local Language Models in Real-World Applications

#### Overview
Semantic search capabilities, enabled by vector databases, allow small local language models (LLMs) to understand and retrieve contextually relevant information. This capability is particularly powerful when building real-world applications, as it enables these models to perform tasks that require deep contextual understanding and relevance matching.

#### Real-World Application Example: Smart Home Automation

Imagine you are building a smart home automation system that uses a small local LLM to control various devices based on user commands. The system needs to understand complex, nuanced commands and retrieve relevant device configurations or actions.

### Step-by-Step Explanation

#### 1. **Embedding Generation**
- **Input**: User command (e.g., "Turn on the living room lights at sunset").
- **Process**: The command is converted into an embedding using a pre-trained model (e.g., BERT, GPT).

```python
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode the user command
inputs = tokenizer("Turn on the living room lights at sunset", return_tensors='pt')

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Get the embedding of the [CLS] token, which represents the whole sentence
embedding = outputs.last_hidden_state[:, 0, :].numpy()
```

#### 2. **Storing Device Configurations in a Vector Database**
- **Input**: Device configurations (e.g., "Living room lights", "Sunset trigger").
- **Process**: Each configuration is converted into an embedding and stored in the vector database.

```python
from annoy import AnnoyIndex

# Define the dimensionality of the embeddings
embedding_dim = 768  # BERT embeddings have 768 dimensions

# Create an Annoy index
index = AnnoyIndex(embedding_dim, 'angular')

# Add device configurations to the index
device_configs = [
    ("Living room lights", embedding1),
    ("Sunset trigger", embedding2),
    # Add more configurations as needed
]

for i, (name, emb) in enumerate(device_configs):
    index.add_item(i, emb)

# Build the index
index.build(10)  # 10 trees for efficient search
```

#### 3. **Semantic Search and Retrieval**
- **Input**: User command embedding.
- **Process**: The vector database performs a semantic search to find the most relevant device configurations.

```python
# Perform a search in the vector database
num_results = 2
similar_items = index.get_nns_by_vector(embedding, num_results, include_distances=True)

# Retrieve the most relevant device configurations
relevant_configs = [device_configs[i][0] for i in similar_items[0]]
```

#### 4. **Action Execution**
- **Input**: Relevant device configurations.
- **Process**: The LLM interprets the retrieved configurations and executes the appropriate actions.

```python
def execute_action(config):
    if "Living room lights" in config:
        turn_on_lights("living_room")
    if "Sunset trigger" in config:
        schedule_action("sunset", turn_on_lights, "living_room")

for config in relevant_configs:
    execute_action(config)
```

### Benefits in Real-World Applications

1. **Contextual Understanding**: The system can understand complex commands like "Turn on the living room lights at sunset" without needing exact keywords.
2. **Improved Relevance**: The vector database ensures that the most relevant device configurations are retrieved, even if the exact keywords are not

### Customizable Embeddings in Vector Databases

#### Overview
Customizable embeddings are a powerful feature of vector databases that allow users to tailor the numerical representations of text (embeddings) to their specific domain or use case. This customization can significantly improve the relevance of search results, making the system more effective and efficient.

#### Why Customizable Embeddings Matter
- **Domain-Specific Knowledge**: Different domains (e.g., healthcare, finance, legal) have unique terminologies and concepts. Custom embeddings can capture these nuances better than generic embeddings.
- **Improved Relevance**: By training embeddings on domain-specific data, the system can retrieve more relevant documents, even for complex queries.
- **Flexibility**: Users can adapt the embeddings to meet the specific needs of their application, ensuring that the system performs optimally in their context.

### Step-by-Step Explanation

#### 1. **Data Collection and Preprocessing**
- **Input**: Domain-specific text data (e.g., medical records, financial reports, legal documents).
- **Process**: Collect and preprocess the data to remove noise and ensure consistency.

```python
import pandas as pd

# Load domain-specific data
data = pd.read_csv('domain_specific_data.csv')

# Preprocess the data (e.g., lowercasing, removing stopwords)
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
```

#### 2. **Training Custom Embeddings**
- **Input**: Preprocessed domain-specific data.
- **Process**: Use a machine learning model to train custom embeddings. Popular models for this purpose include Word2Vec, GloVe, and BERT.

```python
from gensim.models import Word2Vec

# Tokenize the text data
tokenized_data = [text.split() for text in data['text']]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("domain_specific_embeddings.model")
```

#### 3. **Storing Custom Embeddings in a Vector Database**
- **Input**: Custom embeddings.
- **Process**: Convert the text data into custom embeddings and store them in the vector database.

```python
from annoy import AnnoyIndex

# Define the dimensionality of the embeddings
embedding_dim = 100  # Custom embeddings have 100 dimensions

# Create an Annoy index
index = AnnoyIndex(embedding_dim, 'angular')

# Add documents to the index
for i, text in enumerate(data['text']):
    tokens = text.split()
    embedding = sum([model.wv[token] for token in tokens]) / len(tokens)  # Average embedding
    index.add_item(i, embedding)

# Build the index
index.build(10)  # 10 trees for efficient search
```

#### 4. **Semantic Search with Custom Embeddings**
- **Input**: User query.
- **Process**: Convert the query into a custom embedding and perform a semantic search in the vector database.

```python
# Convert user query into custom embedding
query = "diagnosis of heart disease"
query_tokens = query.split()
query_embedding = sum([model.wv[token] for token in query_tokens]) / len(query_tokens)

# Perform a search in the vector database
num_results = 5
similar_items = index.get_nns_by_vector(query_embedding, num_results, include_distances=True)

# Retrieve the most relevant documents
relevant_documents = [data.iloc[i]['text'] for i in similar_items[0]]
```

### Benefits of Customizable Embeddings

1. **Domain-Specific Relevance**: Custom embeddings capture the specific terminology and concepts of the domain, leading to more relevant search results.
2. **Improved Performance**: The system can better understand and retrieve documents that are contextually relevant to the user's query.
3. **Flexibility and Adaptability**: Users can tailor the embeddings to their specific needs, ensuring that the system performs optimally in their context.
4. **Scalability**: The approach can handle large volumes of domain-specific data efficiently, making it scalable for enterprise-level applications.

### Conclusion
Customizable embeddings in vector databases offer significant advantages for improving the relevance of search results in domain-specific applications. By training embeddings on domain-specific data, users can capture the nuances and terminologies of their field, leading to more accurate and contextually appropriate search results. This approach enhances the performance of small local language models, making them more effective in real-world applications.

### Real-World Use Cases of Vector Databases in AI Agent Swarms, NLP Processing, and Local Language Model RAG Software Auto-Complete

#### Overview
Vector databases play a crucial role in enhancing the capabilities of AI agent swarms, NLP processing, and local language model (LLM) RAG software auto-complete systems. By integrating seamlessly with RAG pipelines and enabling dynamic retrieval, vector databases provide a robust foundation for these applications.

### Use Case 1: AI Agent Swarms

#### Scenario: Collaborative Decision-Making in Emergency Response
In an emergency response scenario, multiple AI agents (e.g., medical, logistics, communication) need to collaborate to make informed decisions quickly. Each agent has access to different types of data (e.g., patient records, supply chain information, communication logs).

#### Implementation Steps

1. **Embedding Generation**: Each agent generates embeddings for its data using domain-specific models.
2. **Storing in Vector Database**: The embeddings are stored in a centralized vector database.
3. **Dynamic Retrieval**: When a new emergency arises, the central coordinator queries the vector database to retrieve relevant information from all agents.
4. **Decision-Making**: The retrieved information is used to make informed decisions collaboratively.

```python
# Example code for embedding generation and storage
from transformers import BertTokenizer, BertModel
from annoy import AnnoyIndex

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Store embeddings in vector database
embedding_dim = 768
index = AnnoyIndex(embedding_dim, 'angular')

data = [
    ("Patient record 1", "Patient John Doe, age 35, history of heart disease"),
    ("Supply chain info", "Medical supplies for heart disease treatment"),
    # Add more data as needed
]

for i, (name, text) in enumerate(data):
    embedding = generate_embedding(text)
    index.add_item(i, embedding)

index.build(10)

# Dynamic retrieval
query = "Emergency: heart disease patient"
query_embedding = generate_embedding(query)
similar_items = index.get_nns_by_vector(query_embedding, 5, include_distances=True)
relevant_data = [data[i][1] for i in similar_items[0]]
```

### Use Case 2: NLP Processing

#### Scenario: Sentiment Analysis in Customer Support
A customer support system needs to analyze incoming messages to determine sentiment and provide appropriate responses. The system uses a local LLM for response generation.

#### Implementation Steps

1. **Embedding Generation**: Incoming messages are converted into embeddings.
2. **Storing in Vector Database**: The embeddings are stored in a vector database.
3. **Dynamic Retrieval**: When a new message arrives, the system retrieves similar historical messages to understand context.
4. **Response Generation**: The LLM uses the retrieved context to generate an appropriate response.

```python
# Example code for sentiment analysis and response generation
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Function to analyze sentiment and generate response
def analyze_and_respond(message):
    sentiment = sentiment_analyzer(message)[0]['label']
    query_embedding = generate_embedding(message)
    similar_items = index.get_nns_by_vector(query_embedding, 5, include_distances=True)
    relevant_context = [data[i][1] for i in similar_items[0]]
    
    # Use LLM to generate response based on sentiment and context
    response = llm_generate_response(sentiment, relevant_context)
    return response

# Example usage
message = "I'm having trouble with my order."
response = analyze_and_respond(message)
print(response)
```

### Use Case 3: Local Language Model RAG Software Auto-Complete

#### Scenario: Code Auto-Complete in Integrated Development Environments (IDEs)
An IDE uses a local LLM to provide code auto-complete suggestions based on the context of the code being written.

#### Implementation Steps

1. **Embedding Generation**: Code snippets are converted into embeddings.
2. **Storing in Vector Database**: The embeddings are stored in a vector database.
3. **Dynamic Retrieval**: As the user types, the IDE retrieves similar code snippets to provide context-aware suggestions.
4. **Auto-Complete**: The LLM uses the retrieved context to generate relevant auto-complete suggestions.

```python
# Example code for code auto-complete
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate code suggestions
def generate_code_suggestions(code_snippet):
    query_embedding = generate_embedding(code_snippet)
    similar_items = index.get_nns_by_vector(query_embedding, 5, include_distances=True)
    relevant_code = [data[i][1] for i in similar_items[0]]
    
    # Use LLM to generate code suggestions based on context
    inputs = tokenizer(code_snippet, return_tensors='pt')
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=5)
    suggestions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return suggestions

# Example usage
code_snippet = "def calculate_area(radius):"
suggestions = generate_code_suggestions(code_snippet)
print(suggestions)
```

### Benefits of Vector Databases in These Use Cases

1. **Seamless Integration with RAG Pipelines**: Vector databases ensure a cohesive and efficient workflow by integrating smoothly with embedding generation, document retrieval, and response generation.
2. **Dynamic Retrieval**: The ability to dynamically retrieve and update documents based on user queries allows for a more interactive and responsive user experience.
3. **Improved Relevance**: Customizable embeddings capture domain-specific nuances, leading to more relevant search results and better decision-making.
4. **Scalability**: Vector databases handle large volumes of data efficiently, making them suitable for enterprise-level applications.

### Conclusion
Vector databases are indispensable for enhancing the capabilities of AI agent swarms, NLP processing, and local language model RAG software auto-complete systems. By providing seamless integration with RAG pipelines and enabling dynamic retrieval, vector databases ensure that these applications perform optimally, delivering accurate and contextually appropriate results in real-world scenarios.

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
