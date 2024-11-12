To store the embeddings produced by the OCR pipeline, you can use several types of data structures depending on your requirements for storage, retrieval, and scalability. Here are some common options:

### 1. **In-Memory Data Structures**
- **List of Tensors**: Store embeddings as a list of PyTorch tensors.
  ```python
  embeddings = []
  embeddings.append(image_embedding)
  embeddings.append(text_embedding)
  ```

### 2. **Database Storage**
- **SQL Database**: Store embeddings in a relational database like PostgreSQL. Each embedding can be stored as a binary large object (BLOB).
  ```sql
  CREATE TABLE embeddings (
      id SERIAL PRIMARY KEY,
      pdf_file VARCHAR(255),
      page_number INT,
      embedding BYTEA
  );
  ```

- **NoSQL Database**: Use a NoSQL database like MongoDB to store embeddings as documents.
  ```json
  {
      "pdf_file": "example.pdf",
      "page_number": 1,
      "embedding": [0.1, 0.2, 0.3, ...]
  }
  ```

### 3. **Vector Databases**
- **FAISS**: Use Facebook AI Similarity Search (FAISS) for efficient similarity search.
  ```python
  import faiss
  index = faiss.IndexFlatL2(dimension)
  index.add(np.array(embeddings))
  ```

- **Elasticsearch with Vector Search**: Use Elasticsearch with vector search capabilities.
  ```json
  {
      "mappings": {
          "properties": {
              "pdf_file": { "type": "keyword" },
              "page_number": { "type": "integer" },
              "embedding": { "type": "dense_vector", "dims": 768 }
          }
      }
  }
  ```

- **Pinecone**: Use Pinecone for scalable vector search.
  ```python
  import pinecone
  pinecone.init(api_key='your-api-key')
  index = pinecone.Index("embeddings-index")
  index.upsert(items=[("id1", embedding1), ("id2", embedding2)])
  ```

### Example: Using FAISS for Storing Embeddings
```python
import faiss
import numpy as np

# Assuming embeddings is a list of numpy arrays
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)

# Convert list of embeddings to numpy array
embedding_matrix = np.vstack(embeddings)
index.add(embedding_matrix)

# Save the index to disk
faiss.write_index(index, "embeddings.index")
```

### Example: Using MongoDB for Storing Embeddings
```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["embeddings_db"]
collection = db["embeddings"]

# Assuming embeddings is a list of dictionaries
for embedding in embeddings:
    collection.insert_one(embedding)
```

Choose the data structure that best fits your use case, considering factors like scalability, retrieval speed, and ease of integration with your existing systems.


To expand on in-memory data structures for storing embeddings, we can consider several options depending on the complexity and requirements of the application. Here are some detailed examples:

### 1. **List of Tensors**
A simple and straightforward way to store embeddings in memory is to use a list of PyTorch tensors. This is suitable for small to medium-sized datasets where the entire set of embeddings can fit into memory.

```python
import torch

# Example embeddings
embedding1 = torch.tensor([0.1, 0.2, 0.3])
embedding2 = torch.tensor([0.4, 0.5, 0.6])

# Store embeddings in a list
embeddings = [embedding1, embedding2]

# Accessing an embedding
print(embeddings[0])
```

### 2. **Dictionary of Tensors**
If you need to associate embeddings with specific identifiers (e.g., file names or page numbers), you can use a dictionary where the keys are identifiers and the values are the embeddings.

```python
import torch

# Example embeddings
embedding1 = torch.tensor([0.1, 0.2, 0.3])
embedding2 = torch.tensor([0.4, 0.5, 0.6])

# Store embeddings in a dictionary
embeddings = {
    "file1_page1": embedding1,
    "file1_page2": embedding2
}

# Accessing an embedding
print(embeddings["file1_page1"])
```

### 3. **Numpy Arrays**
For numerical operations and compatibility with various machine learning libraries, you can use numpy arrays to store embeddings.

```python
import numpy as np

# Example embeddings
embedding1 = np.array([0.1, 0.2, 0.3])
embedding2 = np.array([0.4, 0.5, 0.6])

# Store embeddings in a list
embeddings = [embedding1, embedding2]

# Accessing an embedding
print(embeddings[0])
```

### 4. **Pandas DataFrame**
For more complex data manipulation and analysis, you can use a pandas DataFrame to store embeddings along with additional metadata.

```python
import pandas as pd
import numpy as np

# Example embeddings
embedding1 = np.array([0.1, 0.2, 0.3])
embedding2 = np.array([0.4, 0.5, 0.6])

# Create a DataFrame
data = {
    "file": ["file1", "file1"],
    "page": [1, 2],
    "embedding": [embedding1, embedding2]
}
df = pd.DataFrame(data)

# Accessing an embedding
print(df.iloc[0]["embedding"])
```

### 5. **Named Tuples**
For a more structured approach, you can use named tuples to store embeddings along with their associated metadata.

```python
from collections import namedtuple
import torch

# Define a named tuple
Embedding = namedtuple("Embedding", ["file", "page", "tensor"])

# Example embeddings
embedding1 = torch.tensor([0.1, 0.2, 0.3])
embedding2 = torch.tensor([0.4, 0.5, 0.6])

# Store embeddings in a list of named tuples
embeddings = [
    Embedding(file="file1", page=1, tensor=embedding1),
    Embedding(file="file1", page=2, tensor=embedding2)
]

# Accessing an embedding
print(embeddings[0].tensor)
```

### Summary
- **List of Tensors**: Simple and efficient for small datasets.
- **Dictionary of Tensors**: Useful for associating embeddings with identifiers.
- **Numpy Arrays**: Compatible with various numerical operations and libraries.
- **Pandas DataFrame**: Ideal for complex data manipulation and analysis.
- **Named Tuples**: Structured and readable approach for storing embeddings with metadata.

Choose the data structure that best fits your application's requirements in terms of complexity, scalability, and ease of use.
