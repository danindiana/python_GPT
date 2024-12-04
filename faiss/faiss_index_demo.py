import faiss
import numpy as np

# Load the FAISS index using the correct path
index = faiss.read_index("/mnt/nvme1n1/Dec03_embeddings/embeddings.index")

# Check the number of vectors in the index
print("Number of vectors in the index:", index.ntotal)

# Load index
index = faiss.read_index("embeddings.index")
print("Loaded index with vectors:", index.ntotal)

# Query
query_vector = np.random.random((1, index.d)).astype('float32')
distances, indices = index.search(query_vector, 5)
print("Search results:", indices, distances)

# Add new vectors
new_vectors = np.random.random((10, index.d)).astype('float32')
index.add(new_vectors)
print("Updated index with vectors:", index.ntotal)

# Save updated index
faiss.write_index(index, "updated_embeddings.index")
