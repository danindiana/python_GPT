To set up a **batch query API** and **utilize multiple GPUs** effectively, here's a roadmap tailored to your setup:

---

## **1. Batch Query API Setup**
A batch query API allows clients to send multiple queries at once, process them on your FAISS index, and return results in a structured format (e.g., JSON). Below is an example using **FastAPI**:

### **Code for Batch Query API**

#### Install Required Libraries:
```bash
pip install fastapi uvicorn transformers faiss-cpu
```

#### Batch Query API:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import json
import os

# Load FAISS Index and Metadata
output_dir = "/mnt/nvme1n1"
faiss_index_path = os.path.join(output_dir, "large_index.ivf")
text_mapping_path = os.path.join(output_dir, "text_mapping.json")

# Load FAISS Index
if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
index = faiss.read_index(faiss_index_path)

# Load Metadata
if not os.path.exists(text_mapping_path):
    raise FileNotFoundError(f"text_mapping.json not found at {text_mapping_path}")
with open(text_mapping_path, "r") as f:
    text_mapping = json.load(f)

# Load Model and Tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI Initialization
app = FastAPI()

# Define Request Model
class QueryRequest(BaseModel):
    queries: list[str]  # List of query strings
    top_k: int = 5  # Number of results per query

# Helper Function: Generate Query Embeddings
def generate_query_embeddings(queries):
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Helper Function: Search FAISS Index
def search_faiss(embeddings, top_k):
    distances, indices = index.search(embeddings, top_k)
    results = []
    for dist_row, idx_row in zip(distances, indices):
        row_results = []
        for dist, idx in zip(dist_row, idx_row):
            if idx == -1:
                continue  # Skip invalid indices
            result = text_mapping[str(idx)]
            row_results.append({"filename": result["filename"], "distance": float(dist), "snippet": result["text"][:200]})
        results.append(row_results)
    return results

# API Endpoint: Batch Query
@app.post("/batch_query")
async def batch_query(request: QueryRequest):
    try:
        embeddings = generate_query_embeddings(request.queries)
        results = search_faiss(embeddings, request.top_k)
        return {"queries": request.queries, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn filename:app --reload
```

---

### **Running the API**
1. Save the code as `batch_query_api.py`.
2. Start the server:
   ```bash
   uvicorn batch_query_api:app --host 0.0.0.0 --port 8000 --reload
   ```
3. Send a batch query via `curl` or a client like Postman:
   ```bash
   curl -X POST "http://localhost:8000/batch_query" \
        -H "Content-Type: application/json" \
        -d '{"queries": ["Explain neural networks", "How does FAISS work?"], "top_k": 5}'
   ```

---

## **2. Utilizing Multiple GPUs**
To efficiently use multiple GPUs (your RTX 4080 and Quadro P4000), you can distribute FAISS workloads across them.

### **Key FAISS Features for Multi-GPU**
- FAISS provides `StandardGpuResources` for managing GPU memory and `index_cpu_to_all_gpus()` for distributing data across GPUs.

### **Example Code for Multi-GPU FAISS**
```python
import faiss

# Initialize GPU resources
gpu_resources = [faiss.StandardGpuResources() for _ in range(2)]  # Two GPUs
device_map = [0, 1]  # GPU 0 and GPU 1

# Convert CPU FAISS Index to Multi-GPU
gpu_index = faiss.index_cpu_to_all_gpus(index, gpu_resources=gpu_resources, devices=device_map)

# Search with Multi-GPU FAISS Index
query_embedding = generate_query_embeddings(["Example query"])
distances, indices = gpu_index.search(query_embedding, 5)

print("Multi-GPU Search Results:")
for dist, idx in zip(distances[0], indices[0]):
    if idx != -1:
        print(f"Distance: {dist}, Filename: {text_mapping[str(idx)]['filename']}")
```

---

## **Combining the API with Multi-GPU**
- Replace the single-GPU `index` in the API with the multi-GPU `gpu_index`.
- Ensure the batch query embeddings are distributed evenly across GPUs.

---

### **Performance Monitoring**
Use the following tools to monitor GPU and system usage:
- **`nvidia-smi`**: GPU utilization and memory.
- **`htop`**: CPU utilization.

---

Would you like further guidance on deployment (e.g., Dockerizing the API, managing GPUs with Kubernetes)?
