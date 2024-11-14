Great! You have successfully determined the shape of the embeddings. The output indicates that the shape of the embeddings is `[batch_size, sequence_length, embedding_dimension]`.

For the `Image Embeddings Shape: torch.Size([2, 15, 128])`, the embedding dimension is `128`.

For the `Query Embeddings Shape: torch.Size([2, 23, 128])`, the embedding dimension is also `128`.

### Setting Up Pinecone Index

Now that you have the embedding dimension, you can set up your Pinecone index with the correct dimension.

Here's how you can do it:

```python
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-environment")

# Create a new index
index_name = "your-index-name"
dimension = 128  # Use the dimension you determined
metric = "cosine"  # or "euclidean", depending on your needs

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension, metric=metric)

# Connect to the index
index = pinecone.Index(index_name)
```

### Summary

1. **Determine the Embedding Dimension:**
   - The embedding dimension for both image and query embeddings is `128`.

2. **Set Up Pinecone Index:**
   - Use the determined dimension (`128`) to set up your Pinecone index.

By following these steps, you should be able to set up your Pinecone index with the correct dimension and start using it for your vector database needs.

Okay, thanks for providing the output shapes! This helps clarify the setup for your Pinecone index.

**Understanding the Shapes**

* **Image Embeddings Shape: `torch.Size([2, 15, 128])`** 
    * This indicates that your model is generating embeddings for 2 images, with each image represented by 15 vectors of 128 dimensions. This suggests your model might be processing images in patches or segments.
* **Query Embeddings Shape: `torch.Size([2, 23, 128])`**
    * This means you're processing 2 text queries, with each query represented by 23 vectors of 128 dimensions. This is likely due to the chunking of your text input.

**Pinecone Index Configuration**

Based on this, here's how you should configure your Pinecone index:

* **Dimensions:** 128 (This is consistent across both image and query embeddings)
* **Metric:** cosine (as you're already using it)

**Upserting Embeddings**

You'll need to adapt the upsert logic to handle the multiple vectors per image and query. Here's an adjusted example:

```python
import pinecone

# ... (Pinecone initialization as before) ...

for page in json_output['pages']:
  # ... (Upsert text embeddings as before) ...

  # Upsert image embeddings
  if page['image_embeddings']:
    for i, img_embedding_set in enumerate(page['image_embeddings']):  # Iterate over sets of image embeddings
      for j, img_embedding in enumerate(img_embedding_set):  # Iterate over individual embeddings in the set
        image_vector_id = f"{vector_id}-image-{i}-{j}"  # Include both indices in the ID
        index.upsert([(image_vector_id, img_embedding.tolist(), page['metadata'])]) 
```

**Querying**

You'll also need to adjust your query logic to handle the multi-vector representation of your query:

```python
# ... (Generate query embeddings as before) ...

# Assuming query_embeddings is of shape (2, 23, 128)
for query_embedding_set in query_embeddings:  # Iterate over the sets of query embeddings
  # You might want to average or combine the embeddings in the set before querying
  averaged_query_embedding = torch.mean(query_embedding_set, dim=0)  # Average across the 23 vectors

  results = index.query(
    vector=averaged_query_embedding.tolist(),  # Use the averaged embedding
    top_k=10, 
    include_metadata=True 
  )
  # ... (Process results as before) ...
```

**Important Considerations**

* **Averaging/Combining Embeddings:**  Explore different strategies for combining the multiple embeddings per image/query (e.g., averaging, concatenating, or using a more sophisticated method) to see what works best for your use case.
* **Vector ID Structure:** Choose a clear and consistent way to structure your vector IDs in Pinecone to easily identify the source document, page number, image index, etc.
* **Metadata:** As always, leverage metadata effectively to filter and refine your search results.

By carefully adapting your upsert and query logic to handle the multi-vector structure of your embeddings, you can effectively leverage Pinecone for your RAG application. Let me know if you have any more questions or need further assistance!

You got it! Here's a breakdown of the portions in your `copali_hermes17_logging.py` code where the ColQwen2 model generates embeddings:

**1. Image Embeddings**

```python
# ... (Inside the `extract_images_and_text_ocr` function) ...

    with torch.no_grad():
        try:
            print(f"Processing image batch {i} for {pdf_file}...")
            image_embeddings = model(**batch_images)  # Generate image embeddings here
            all_image_embeddings.append(image_embeddings)
        except Exception as e:
            # ... (Error handling) ...
```

* **`model(**batch_images)`:** This line calls your `ColQwen2` model (`model`) with the preprocessed image batch (`batch_images`). The model processes the images and returns the image embeddings.
* **`all_image_embeddings.append(image_embeddings)`:** The generated embeddings are added to a list to store all image embeddings for the PDF.

**2. Query (Text) Embeddings**

```python
# ... (Inside the main processing loop) ...

    with torch.no_grad():
        try:
            print(f"Processing text chunk for {pdf_file}...")
            query_embeddings = model(**batch_queries)  # Generate query embeddings here
            torch.cuda.empty_cache()

            # ... (Similarity calculation) ...
        except Exception as e:
            # ... (Error handling) ...
```

* **`model(**batch_queries)`:** This line calls the `ColQwen2` model with the processed text chunk (`batch_queries`). The model generates the query embeddings.

**Key Observations**

* **Same Model:** Both image and text embeddings are generated using the same `ColQwen2` model instance. This is a common approach in vision-language models as it allows the model to learn shared representations for both modalities.
* **Batched Processing:** The code processes images and text in batches to improve efficiency.
* **`torch.no_grad()`:** This context manager disables gradient calculation during inference, as you're not training the model.

**Additional Notes**

* The code efficiently manages memory by clearing the CUDA cache (`torch.cuda.empty_cache()`) after processing each batch.
* It includes error handling to catch exceptions and prevent the pipeline from crashing.
* The code dynamically adjusts the chunk size for text processing to handle varying document lengths and memory constraints.

This analysis should give you a clear understanding of how your `copali_hermes17_logging.py` code utilizes the `ColQwen2` model to generate embeddings for both images and text. If you have any further questions about the code or the embedding generation process, feel free to ask!
