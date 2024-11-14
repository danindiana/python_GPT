import os
import json
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

def split_text_into_chunks(text, chunk_size):
    """Split text into chunks of the specified size."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_embeddings(text_chunks, model, processor):
    """Generate embeddings for the given text chunks."""
    embeddings = []
    for chunk in text_chunks:
        queries = [chunk]
        batch_queries = processor.process_queries(queries).to(device)
        with torch.no_grad():
            chunk_embeddings = model(**batch_queries)
            embeddings.append(chunk_embeddings.tolist())
    return embeddings

def binary_quantize(embeddings):
    """Convert float32 embeddings to binary (1-bit) values."""
    binary_embeddings = []
    for embedding in embeddings:
        binary_embedding = [1 if val > 0 else 0 for val in embedding]
        binary_embeddings.append(binary_embedding)
    return binary_embeddings

def hamming_distance(binary_embedding1, binary_embedding2):
    """Calculate the Hamming Distance between two binary embeddings."""
    return sum(bit1 != bit2 for bit1, bit2 in zip(binary_embedding1, binary_embedding2))

def rescore(float32_query_embedding, binary_document_embeddings, top_k, rescore_multiplier):
    """Rescore the top-k binary document embeddings with the float32 query embedding."""
    # Convert float32 query embedding to binary
    binary_query_embedding = binary_quantize([float32_query_embedding])[0]
    
    # Calculate Hamming Distances
    distances = [hamming_distance(binary_query_embedding, binary_doc_emb) for binary_doc_emb in binary_document_embeddings]
    
    # Get top-k results based on Hamming Distance
    top_k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:top_k * rescore_multiplier]
    
    # Rescore the top-k binary document embeddings with the float32 query embedding
    rescore_results = []
    for idx in top_k_indices:
        binary_doc_emb = binary_document_embeddings[idx]
        # Convert binary document embedding back to float32 for rescoring
        float32_doc_emb = [val * 2 - 1 for val in binary_doc_emb]
        score = torch.dot(torch.tensor(float32_query_embedding), torch.tensor(float32_doc_emb)).item()
        rescore_results.append((idx, score))
    
    # Sort by rescore scores and return top-k results
    rescore_results.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in rescore_results[:top_k]]

def process_text_file(filepath, model, processor, max_chunk_size=5000, max_sequence_length=32768, top_k=10, rescore_multiplier=2):
    """Processes a single text file and returns the JSON output."""

    with open(filepath, "r") as f:
        text = f.read()

    # Process text and generate embeddings
    text_chunks = split_text_into_chunks(text, max_chunk_size)
    float32_embeddings = generate_embeddings(text_chunks, model, processor)
    
    # Convert float32 embeddings to binary
    binary_embeddings = binary_quantize(float32_embeddings)

    # Prepare JSON output
    json_output = {
        "text_filename": os.path.basename(filepath),
        "chunks": []
    }

    for i, chunk in enumerate(text_chunks):
        chunk_data = {
            "chunk_number": i + 1,
            "text": chunk,
            "float32_embedding": float32_embeddings[i],
            "binary_embedding": binary_embeddings[i]
        }
        json_output["chunks"].append(chunk_data)

    return json_output, binary_embeddings

# Ask the user for input and output directories
input_text_dir = input("Enter the path of the directory containing text files: ")
output_json_dir = input("Enter the path of the output directory for JSON files: ")

# Verify the directories exist
if not os.path.isdir(input_text_dir):
    print("The text directory does not exist.")
    exit()
if not os.path.isdir(output_json_dir):
    print("The output directory does not exist.")
    exit()

# Load model and processor only after directory confirmation to delay GPU allocation
device = torch.device("cuda:0")
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.float16  # Ensure half-precision to save memory
).to(device).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Set a lower maximum chunk size for memory efficiency
max_chunk_size = 5000  # Reduced to 5000 to avoid high memory usage
max_sequence_length = 32768  # Define the max sequence length
top_k = 10  # Number of top results to retrieve
rescore_multiplier = 2  # Multiplier for rescoring step

# Process all text files in the target directory
text_files = [f for f in os.listdir(input_text_dir) if f.endswith('.txt')]

if not text_files:
    print("No text files found in the specified directory.")
    exit()

# Process each text file in the input directory
for text_file in text_files:
    filepath = os.path.join(input_text_dir, text_file)
    try:
        json_output, binary_embeddings = process_text_file(filepath, model, processor)

        # Save JSON output to a file
        output_json_file = os.path.join(output_json_dir, f"{text_file[:-4]}_embeddings.json")
        with open(output_json_file, "w") as f:
            json.dump(json_output, f, indent=4)

        print(f"Embeddings and metadata saved to {output_json_file}")

    except Exception as e:
        print(f"Error processing {text_file}: {e}")

# Final memory cleanup
torch.cuda.empty_cache()
