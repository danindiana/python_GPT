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

Pinecone is a managed vector database service that runs in the cloud, which means it requires an API key to authenticate and interact with its services. This API key is used to securely access your Pinecone account and manage your indexes and data.

### Why Pinecone Requires an API Key
1. **Authentication**: The API key ensures that only authorized users can access and modify your data.
2. **Security**: It provides a secure way to interact with the Pinecone service over the internet.
3. **Resource Management**: Pinecone runs on cloud infrastructure, and the API key helps manage and allocate resources for your account.

### Local Alternatives to Pinecone
If you prefer to run a vector search engine locally without requiring an API key, you can consider the following alternatives:

1. **FAISS (Facebook AI Similarity Search)**:
   - FAISS is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors.
   - It can be run locally and does not require an API key.

   ```python
   import faiss
   import numpy as np

   # Example embeddings
   embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

   # Create a FAISS index
   dimension = embeddings.shape[1]
   index = faiss.IndexFlatL2(dimension)

   # Add embeddings to the index
   index.add(embeddings)

   # Query the index
   query_embedding = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
   distances, indices = index.search(query_embedding, k=2)

   print("Distances:", distances)
   print("Indices:", indices)
   ```

2. **Annoy (Approximate Nearest Neighbors Oh Yeah)**:
   - Annoy is a library developed by Spotify for approximate nearest neighbor search.
   - It can be run locally and does not require an API key.

   ```python
   from annoy import AnnoyIndex

   # Example embeddings
   dimension = 3
   index = AnnoyIndex(dimension, 'euclidean')

   # Add embeddings to the index
   index.add_item(0, [0.1, 0.2, 0.3])
   index.add_item(1, [0.4, 0.5, 0.6])

   # Build the index
   index.build(10)  # 10 trees

   # Query the index
   query_embedding = [0.1, 0.2, 0.3]
   indices = index.get_nns_by_vector(query_embedding, 2)

   print("Indices:", indices)
   ```

3. **Elasticsearch with Vector Search**:
   - Elasticsearch is a distributed search engine that supports vector search through plugins.
   - It can be run locally but requires more setup compared to FAISS and Annoy.

   ```json
   # Example Elasticsearch mapping for vector search
   {
       "mappings": {
           "properties": {
               "embedding": {
                   "type": "dense_vector",
                   "dims": 3
               }
           }
       }
   }
   ```

### Summary
- **Pinecone**: Requires an API key because it is a managed cloud service.
- **Local Alternatives**: FAISS, Annoy, and Elasticsearch can be run locally without an API key.

Choose the solution that best fits your requirements for scalability, ease of use, and deployment.

Pinecone is a managed vector database service that offers several advantages, making it an attractive option for certain use cases. Here are some reasons why someone might choose Pinecone:

### 1. **Scalability**
- **Automatic Scaling**: Pinecone automatically scales to handle large volumes of data and high query loads without requiring manual intervention.
- **Elastic Resources**: It can dynamically allocate resources based on the workload, ensuring optimal performance.

### 2. **Ease of Use**
- **Managed Service**: As a fully managed service, Pinecone handles infrastructure management, including hardware provisioning, software updates, and maintenance.
- **Simple API**: Pinecone provides a straightforward API for creating, updating, and querying indexes, making it easy to integrate into applications.

### 3. **Performance**
- **Low Latency**: Pinecone is optimized for low-latency vector search, providing fast query responses even with large datasets.
- **High Throughput**: It supports high-throughput operations, allowing for efficient handling of numerous queries and updates.

### 4. **Advanced Features**
- **Hybrid Search**: Pinecone supports hybrid search, combining vector search with traditional keyword search for more comprehensive results.
- **Metadata Filtering**: It allows for filtering results based on metadata, enabling more precise and relevant search outcomes.

### 5. **Security**
- **Data Encryption**: Pinecone ensures data security with encryption at rest and in transit.
- **Access Control**: It provides robust access control mechanisms to manage permissions and secure data access.

### 6. **Integration**
- **Compatibility**: Pinecone integrates seamlessly with popular machine learning frameworks and libraries, such as TensorFlow, PyTorch, and Hugging Face.
- **Ecosystem**: It fits well into the broader ecosystem of data processing and machine learning tools, facilitating end-to-end workflows.

### 7. **Support and Reliability**
- **SLA**: Pinecone offers service level agreements (SLAs) that guarantee uptime and reliability.
- **Customer Support**: It provides dedicated customer support to assist with any issues or questions.

### Example Use Cases
- **Recommendation Systems**: Pinecone can be used to build recommendation systems that require fast and accurate similarity search.
- **Semantic Search**: It is ideal for applications that need to perform semantic search over large text corpora or document collections.
- **Image Retrieval**: Pinecone can efficiently handle image retrieval tasks by storing and querying image embeddings.

### Summary
Pinecone is a powerful and convenient option for scalable vector search due to its managed nature, ease of use, performance, advanced features, security, and integration capabilities. It is particularly suitable for applications that require efficient and reliable similarity search over large datasets.

Here is an example of how to integrate Pinecone with the existing OCR embedding output. This example assumes you have already extracted embeddings from your OCR pipeline and now want to store and query these embeddings using Pinecone.

### Step-by-Step Integration

#### 1. **Install Pinecone Client**
First, install the Pinecone client library:
```bash
pip install pinecone-client
```

#### 2. **Initialize Pinecone**
Initialize the Pinecone client with your API key:
```python


import

 pinecone

# Initialize Pinecone
pinecone.init(api_key='your-api-key')
```

#### 3. **Create an Index**
Create an index to store your embeddings. Specify the dimension of the embeddings and the metric for similarity search (e.g., cosine similarity):
```python
# Create an index
index_name = "ocr-embeddings-index"
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

### Full Example Integration with OCR Pipeline

Here is a full example integrating Pinecone with the OCR pipeline:

```python
import pinecone
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
import numpy as np

# Initialize Pinecone
pinecone.init(api_key='your-api-key')
index_name = "ocr-embeddings-index"
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
- **Install Pinecone Client**: Install the Pinecone client library.
- **Initialize Pinecone**: Initialize the Pinecone client with your API key.
- **Create an Index**: Create an index to store embeddings.
- **Upsert Embeddings**: Insert or update embeddings in the Pinecone index.
- **Query the Index**: Retrieve the most similar embeddings to a given query embedding.

This setup allows you to efficiently store and retrieve embeddings, enabling scalable vector search for your OCR pipeline.

The dimension of the embeddings and the metric for similarity are determined by the model you are using to generate the embeddings. Here’s how you can determine these parameters:

### Determining the Dimension of Embeddings

1. **Model Documentation**:
   - Check the documentation of the model you are using (e.g., `ColQwen2`). The documentation typically specifies the dimension of the output embeddings.
   - For example, if you are using a pre-trained model from Hugging Face, the model card will often include the embedding dimension.

2. **Inspecting the Model Output**:
   - You can inspect the shape of the embeddings generated by the model to determine the dimension.
   - Example in PyTorch:
     ```python
     import torch
     from colpali_engine.models import ColQwen2, ColQwen2Processor

     # Load model and processor
     device = torch.device("cuda:0")
     model = ColQwen2.from_pretrained("vidore/colqwen2-v0.1", torch_dtype=torch.float16).to(device).eval()
     processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

     # Example input
     example_input = processor.process_images([example_image]).to(device)

     # Generate embeddings
     with torch.no_grad():
         embeddings = model(**example_input)

     # Print the shape of the embeddings
     print(embeddings.shape)  # Output: torch.Size([batch_size, dimension])
     ```

### Choosing the Metric for Similarity

1. **Cosine Similarity**:
   - Cosine similarity is commonly used for high-dimensional embeddings, especially when the magnitude of the vectors is not as important as their direction.
   - It measures the cosine of the angle between two vectors, providing a similarity score between -1 and 1.

2. **Euclidean Distance**:
   - Euclidean distance is another common metric, which measures the straight-line distance between two vectors in the embedding space.
   - It is suitable when the magnitude of the vectors is important.

3. **Dot Product**:
   - The dot product can be used as a similarity metric, especially in cases where the magnitude of the vectors is meaningful.

### Example: Determining Dimension and Metric

Assuming you are using the `ColQwen2` model from the example:

1. **Determine Dimension**:
   ```python
   import torch
   from colpali_engine.models import ColQwen2, ColQwen2Processor

   # Load model and processor
   device = torch.device("cuda:0")
   model = ColQwen2.from_pretrained("vidore/colqwen2-v0.1", torch_dtype=torch.float16).to(device).eval()
   processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

   # Example input
   example_image = ...  # Load or generate an example image
   example_input = processor.process_images([example_image]).to(device)

   # Generate embeddings
   with torch.no_grad():
       embeddings = model(**example_input)

   # Print the shape of the embeddings
   print(embeddings.shape)  # Output: torch.Size([batch_size, dimension])
   ```

   The second value in the shape tuple is the dimension of the embeddings.

2. **Choose Metric**:
   - For most text and image embeddings, cosine similarity is a good default choice.
   - If you have specific requirements or domain knowledge suggesting another metric, you can choose accordingly.

### Example: Creating Pinecone Index with Determined Parameters

```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key='your-api-key')

# Determine the dimension from the model output
dimension = 768  # Example dimension, replace with actual dimension from model

# Create an index with cosine similarity
index_name = "ocr-embeddings-index"
pinecone.create_index(index_name, dimension=dimension, metric='cosine')
```

By following these steps, you can accurately determine the dimension of your embeddings and choose an appropriate similarity metric for your Pinecone index.
