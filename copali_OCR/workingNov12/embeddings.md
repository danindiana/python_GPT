The code processes PDF files to extract text and images, performs OCR on the images if necessary, and generates embeddings for the images using a pre-trained model. Here’s a step-by-step explanation of how embeddings are produced and how the output can be used for a Retrieval-Augmented Generator (RAG):

### Producing Embeddings

1. **Extract Images and Text**:
   - The `extract_images_and_text_ocr` function extracts images and text from the PDF. If direct text extraction fails, it uses OCR to extract text from the images.

2. **Process Images**:
   - Images are processed in batches (batch size of 1 to avoid out-of-memory errors).
   - Each image is processed using the `ColQwen2Processor` to prepare it for the model.
   - The processed images are passed through the `ColQwen2` model to generate embeddings.
   - The embeddings are concatenated to form a single tensor representing all image embeddings.

3. **Process Text**:
   - Text is split into manageable chunks.
   - Each chunk is processed using the `ColQwen2Processor` to prepare it for the model.
   - The processed text chunks are passed through the `ColQwen2` model to generate text embeddings.

### Using the Output for RAG

1. **Store Embeddings**:
   - Store the generated image and text embeddings in a database or a file system for later retrieval.

2. **Indexing**:
   - Index the embeddings using a vector search engine like FAISS, Elasticsearch, or Pinecone to enable efficient retrieval.

3. **Retrieval**:
   - When a query is made, convert the query into an embedding using the same model.
   - Use the vector search engine to find the most similar embeddings to the query.

4. **Augmentation**:
   - Retrieve the original text or images corresponding to the similar embeddings.
   - Use this retrieved information to augment the input for a generative model.

5. **Generation**:
   - Pass the augmented input to a generative model (e.g., GPT-3, T5) to produce a response that is informed by the retrieved context.

### Example Workflow for RAG

1. **Query Embedding**:
   - Convert the user query into an embedding using the `ColQwen2Processor` and `ColQwen2` model.

2. **Retrieve Relevant Context**:
   - Use the query embedding to retrieve the most relevant text/image embeddings from the indexed database.

3. **Augment Query**:
   - Combine the retrieved context with the original query to form an augmented input.

4. **Generate Response**:
   - Pass the augmented input to a generative model to produce a response.

This approach leverages the strengths of both retrieval and generation, providing more accurate and contextually relevant responses.

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

pdf_path = "example.pdf"

To expand on using Pinecone for scalable vector search, here is a detailed guide on how to integrate Pinecone with your OCR pipeline to store and retrieve embeddings efficiently.

### Step-by-Step Guide

#### 1. **Install Pinecone**
First, you need to install the Pinecone client library:
```bash
pip install pinecone-client
```

#### 2. **Initialize Pinecone**
Initialize the Pinecone client with your API key:
```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key

='

your-api-key')
```

#### 3. **Create an Index**
Create an index to store your embeddings. You can specify the dimension of the embeddings and the metric for similarity search (e.g., cosine similarity):
```python
# Create an index
index_name = "embeddings-index"
dimension = 768  # Example dimension, adjust based on your embeddings
pinecone.create_index(index_name, dimension=dimension, metric='cosine')
```

#### 4. **Upsert Embeddings**
Upsert (insert or update) embeddings into the Pinecone index. Each embedding should have a unique ID:
```python
# Connect to the index
index = pinecone.Index(index_name)

# Example embeddings and metadata
embeddings = [
    {"id": "file1_page1", "values": [0.1, 0.2, 0.3, ...]},
    {"id": "file1_page2", "values": [0.4, 0.5, 0.6, ...]}
]

# Upsert embeddings
index.upsert(items=embeddings)
```

#### 5. **Query the Index**
Query the index to retrieve the most similar embeddings to a given query embedding:
```python
# Example query embedding
query_embedding = [0.1, 0.2, 0.3, ...]

# Query the index
results = index.query(queries=[query_embedding], top_k=5)

# Print the results
for result in results['matches']:
    print(f"ID: {result['id']}, Score: {result['score']}")
```

#### 6. **Delete the Index (Optional)**
If you need to delete the index, you can do so with the following command:
```python
pinecone.delete_index(index_name)
```

### Full Example Integration

Here is a full example integrating Pinecone with the OCR pipeline:

```python
import pinecone
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Initialize Pinecone
pinecone.init(api_key='your-api-key')
index_name = "embeddings-index"
dimension = 768  # Example dimension, adjust based on your embeddings
pinecone.create_index(index_name, dimension=dimension, metric='cosine')
index = pinecone.Index(index_name)

# Load model and processor
device = torch.device("cuda:0")
model = ColQwen2.from_pretrained("vidore/colqwen2-v0.1", torch_dtype=torch.float16).to(device).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Example function to process and upsert embeddings
def process_and_upsert(pdf_path):
    images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)
    
    # Process images
    all_image_embeddings = []
    if images:
        for i in range(0, len(images), 1):  # Batch size reduced to 1
            image_batch = images[i:i + 1]
            batch_images = processor.process_images(image_batch).to(device)
            
            with torch.no_grad():
                image_embeddings = model(**batch_images)
                all_image_embeddings.append(image_embeddings.cpu().numpy())
            
            torch.cuda.empty_cache()

        if all_image_embeddings:
            all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
        else:
            all_image_embeddings = None

    # Upsert image embeddings to Pinecone
    if all_image_embeddings is not None:
        for idx, embedding in enumerate(all_image_embeddings):
            item = {"id": f"{pdf_path}_page{idx+1}", "values": embedding.tolist()}
            index.upsert(items=[item])

# Example usage
pdf_path = "example.pdf"
process_and_upsert(pdf_path)

# Query the index
query_embedding = [0.1, 0.2, 0.3, ...]  # Example query embedding
results = index.query(queries=[query_embedding], top_k=5)

# Print the results
for result in results['matches']:
    print(f"ID: {result['id']}, Score: {result['score']}")
```

### Summary
- **Install Pinecone**: Install the Pinecone client library.
- **Initialize Pinecone**: Initialize the Pinecone client with your API key.
- **Create an Index**: Create an index to store embeddings.
- **Upsert Embeddings**: Insert or update embeddings in the Pinecone index.
- **Query the Index**: Retrieve the most similar embeddings to a given query embedding.
- **Delete the Index**: Optionally, delete the index if no longer needed.

This setup allows you to efficiently store and retrieve embeddings, enabling scalable vector search for your OCR pipeline.
