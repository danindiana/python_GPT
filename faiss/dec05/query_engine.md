Based on your embeddings generation script, a query script will leverage the generated FAISS index and `text_mapping` JSON file to search for the most similar embeddings given a query. Here's an example:

### Query Script

```python
import os
import numpy as np
import faiss
import json
from transformers import AutoTokenizer, AutoModel
import torch

# Define the paths to the FAISS index and text_mapping.json
output_dir = '/mnt/nvme1n1/Dec05_embeddings'
faiss_index_path = os.path.join(output_dir, 'embeddings.index')
text_mapping_path = os.path.join(output_dir, 'text_mapping.json')

# Load the FAISS index
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
index = faiss.read_index(faiss_index_path)

# Load the text mapping
if not os.path.exists(text_mapping_path):
    raise FileNotFoundError(f"text_mapping.json not found at {text_mapping_path}")
with open(text_mapping_path, 'r') as f:
    text_mapping = json.load(f)

# Load the pre-trained model and tokenizer (same as in the embeddings script)
model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate embeddings for a query
def generate_query_embedding(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return query_embedding

# Function to search the FAISS index
def search_index(query, top_k=5):
    query_embedding = generate_query_embedding(query)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

# Main query loop
while True:
    query = input("Enter your search query (or 'exit' to quit): ").strip()
    if query.lower() == 'exit':
        print("Exiting...")
        break

    # Search the FAISS index
    distances, indices = search_index(query, top_k=5)

    # Display results
    print("\nSearch Results:")
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx == -1:  # No more results
            break
        result = text_mapping[str(idx)]
        print(f"{i + 1}. Filename: {result['filename']} (Distance: {dist:.4f})")
        print(f"   Snippet: {result['text'][:200]}...")  # Display the first 200 characters
```

### Key Components of the Script
1. **Load the FAISS Index and Metadata**:
   - Reads the FAISS index and `text_mapping.json` that were saved during embedding generation.

2. **Generate Query Embeddings**:
   - Uses the same model and tokenizer to create embeddings for the user's query.

3. **Search the FAISS Index**:
   - Searches for the `top_k` nearest embeddings in the FAISS index.

4. **Display Results**:
   - Uses the indices to retrieve filenames and snippets of text from the `text_mapping`.

### Usage
1. Place this script in the same directory as your embedding generation script.
2. Run the script and input your query to find similar documents.

Let me know if you'd like adjustments or additional features!
