OK, here's how we can modify your `copali_hermes17_logging.py` to output embeddings in a format suitable for an open-source vector database and a good choice for the storage format:

**1. Choose an Open-Source Vector Database**

For this example, let's use **pgvector**, which extends PostgreSQL with vector search capabilities. It's a good choice because:

* **Simplicity:** Easy to set up and integrate with your existing PostgreSQL infrastructure.
* **SQL Integration:**  Leverage the power and familiarity of SQL for managing and querying your data.
* **Performance:**  Offers good performance for many use cases.

**2. Modify `copali_hermes17_logging.py` for JSON Output**

We'll adjust your code to produce JSON output containing the embeddings and metadata. Here's the modified code:

```python
import os
import json
import pytesseract
import fitz  # PyMuPDF
import torch
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
from colpali_engine.models import ColQwen2, ColQwen2Processor

# ... (rest of your imports and functions remain the same) ...


def extract_metadata(pdf_path):
    """Extract metadata from the PDF (replace with your actual logic)."""
    metadata = {}
    try:
        with fitz.open(pdf_path) as doc:
            metadata["author"] = doc.metadata.get("author", "")
            metadata["creationDate"] = doc.metadata.get("creationDate", "")
            # ... extract other relevant metadata ...
    except Exception as e:
        print(f"Failed to extract metadata from {pdf_path}: {e}")
    return metadata

# ... (rest of your functions remain the same) ...

# Process each PDF file in the input directory
for pdf_file in pdf_files:
    # ... (file processing logic remains the same) ...

    # ... (embedding generation remains the same) ...

    # Prepare JSON output
    json_output = {
        "pdf_filename": pdf_file,
        "pages": []
    }

    for i, page_embeddings in enumerate(query_embeddings):  # Iterate over pages
        page_data = {
            "page_number": i + 1,
            "text_embeddings": [emb.tolist() for emb in page_embeddings],  # Convert to lists
            "image_embeddings": [[emb.tolist() for emb in emb_set] for emb_set in all_image_embeddings[i]] if all_image_embeddings else [],
            "metadata": extract_metadata(pdf_path)  # Call the metadata extraction function
        }
        json_output["pages"].append(page_data)

    # Save JSON output to a file
    output_json_file = os.path.join(output_dir, f"{pdf_file[:-4]}_embeddings.json")
    with open(output_json_file, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"Embeddings and metadata saved to {output_json_file}")

# ... (rest of the code remains the same) ...
```

**Key Changes**

* **`extract_metadata()` function:**  This is a placeholder. Implement your logic to extract relevant metadata (author, date, etc.) from the PDF using libraries like `fitz`.
* **JSON structure:** The code now generates JSON output with the following structure:
    ```json
    {
      "pdf_filename": "my_document.pdf",
      "pages": [
        {
          "page_number": 1,
          "text_embeddings": [[0.12, 0.34, ...], [0.56, 0.78, ...], ...],  # List of text embeddings for the page
          "image_embeddings": [ 
            [[0.23, 0.45, ...], [0.67, 0.89, ...], ...],  # List of image embedding sets
            [[0.91, 0.12, ...], [0.34, 0.56, ...], ...]   # Each set is a list of embeddings
          ],
          "metadata": {
            "author": "John Doe",
            "date": "2024-11-13"
          }
        },
        # ... more pages ...
      ]
    }
    ```
* **Converting to lists:** The code converts PyTorch tensors to lists using `.tolist()` for compatibility with JSON serialization.

**3. Storage Format: pgvector**

* **PostgreSQL with pgvector:** Store your data in a PostgreSQL database with the `pgvector` extension enabled.
* **Table schema:**
    ```sql
    CREATE TABLE pdf_embeddings (
        id SERIAL PRIMARY KEY,
        pdf_filename TEXT,
        page_number INTEGER,
        text_embedding VECTOR(128),  -- Adjust dimensionality if needed
        image_embedding VECTOR(128),  -- Can be an array of vectors
        metadata JSONB
    );
    ```
* **Data insertion:** Use SQL `INSERT` statements to insert the data from your JSON output into the table. You can use the `pgvector` functions (e.g., `l2_distance`, `cosine_distance`) for similarity searches.

**4. Integration**

You'll need to write a script or modify your code to:

* Read the JSON output from `copali_hermes17_logging.py`.
* Connect to your PostgreSQL database.
* Insert the data into the `pdf_embeddings` table.

**Example using psycopg2 (a PostgreSQL adapter for Python)**

```python
import psycopg2
import json

# ... (database connection details) ...

conn = psycopg2.connect(database="your_database", user="your_user", 
                        password="your_password", host="your_host")
cursor = conn.cursor()

with open("output_embeddings.json", "r") as f:
    data = json.load(f)

for page in data["pages"]:
    cursor.execute("""
        INSERT INTO pdf_embeddings (pdf_filename, page_number, text_embedding, image_embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        data["pdf_filename"],
        page["page_number"],
        page["text_embeddings"][0],  # Assuming you're using the first text embedding for now
        page["image_embeddings"],
        json.dumps(page["metadata"])
    ))

conn.commit()
cursor.close()
conn.close()
```

This modified code generates JSON output that can be easily integrated with a PostgreSQL database using the `pgvector` extension. Remember to adapt the database interaction code to your specific needs and error handling.
