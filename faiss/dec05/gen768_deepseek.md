To modify the code to use the `all-mpnet-base-v2` model instead of `all-MiniLM-L6-v2`, you need to update the model name and ensure that the tokenizer and model are loaded correctly. Additionally, you should check the embedding dimension after loading the model to verify that it matches the expected dimensions of the `all-mpnet-base-v2` model (which is 768).

Here is the modified code:

```python
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import signal
import pickle
import json

# Define the directory containing the text files
text_dir = '/mnt/nvme1n1/Dec03_text_tranch'

# Default output directory for embeddings
default_output_dir = '/mnt/nvme1n1/Dec05_embeddings'

# Prompt user for output directory
output_dir = input(f"Enter the directory to store embeddings (default: {default_output_dir}): ")
output_dir = output_dir.strip() or default_output_dir

# Check if the directory exists and create it if necessary
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    except PermissionError:
        print(f"Error: No write permissions for directory {output_dir}. Exiting.")
        exit(1)

# Check for write permissions with a write test
test_file_path = os.path.join(output_dir, 'test_write.txt')
try:
    with open(test_file_path, 'w') as test_file:
        test_file.write('Test write successful')
    os.remove(test_file_path)  # Clean up the test file
except PermissionError:
    print(f"Error: No write permissions for directory {output_dir}. Exiting.")
    exit(1)

# Prompt user for GPU selection
gpu_id = input("Enter the GPU ID to use (default: 0): ")
gpu_id = int(gpu_id.strip() or 0)

# Set the device to the selected GPU
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model and tokenizer
model_name = 'sentence-transformers/all-mpnet-base-v2'  # 768-dimensional embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move the model to the selected GPU
model.to(device)

# Check the embedding dimension after loading
embedding_dim = model.config.hidden_size
print(f"Embedding dimension: {embedding_dim}")

# Function to generate embeddings
def generate_embeddings(texts):
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling to get the sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move embeddings back to CPU
    return embeddings

# Function to handle graceful shutdown
def handle_shutdown(signum, frame):
    print("\nGraceful shutdown initiated. Saving current state...")
    save_state()
    print("State saved. Exiting.")
    exit(0)

# Function to save the current state
def save_state():
    state = {
        'texts': texts,
        'embeddings': embeddings,
        'index': index,
        'text_mapping': text_mapping
    }
    state_path = os.path.join(output_dir, 'state.pkl')
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"State saved to {state_path}")

# Register the signal handler for graceful shutdown
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Check if there is a saved state to resume from
resume = input("Do you want to resume from a saved state? (yes/no, default: no): ").strip().lower()
if resume == 'yes':
    state_path = os.path.join(output_dir, 'state.pkl')
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            texts = state['texts']
            embeddings = state['embeddings']
            index = state['index']
            text_mapping = state['text_mapping']
        print("Resuming from saved state.")
    else:
        print("No saved state found. Starting from scratch.")
else:
    texts = []
    embeddings = []
    index = None
    text_mapping = {}

# Read all text files and generate embeddings in batches
batch_size = 532  # Adjust batch size based on your GPU memory

for idx, filename in enumerate(os.listdir(text_dir)):
    if filename.endswith('.txt'):
        print(f"Reading file: {filename}")
        with open(os.path.join(text_dir, filename), 'r') as file:
            text = file.read()
            texts.append(text)
            text_mapping[idx] = {"filename": filename, "text": text}  # Store filename and text

        # Process in batches
        if len(texts) >= batch_size:
            print(f"Generating embeddings for batch of {len(texts)} texts...")
            batch_embeddings = generate_embeddings(texts)
            embeddings.append(batch_embeddings)
            texts = []  # Clear the batch

            # Save state periodically
            save_state()

# Process any remaining texts
if texts:
    print(f"Generating embeddings for final batch of {len(texts)} texts...")
    batch_embeddings = generate_embeddings(texts)
    embeddings.append(batch_embeddings)

# Concatenate all embeddings
embeddings = np.vstack(embeddings)

# Create a FAISS index if not already created
if index is None:
    d = embeddings.shape[1]  # dimensionality of the embeddings
    index = faiss.IndexFlatL2(d)  # L2 distance index

# Add embeddings to the FAISS index
index.add(embeddings)

# Save text_mapping to a JSON file
with open(os.path.join(output_dir, 'text_mapping.json'), 'w') as f:
    json.dump(text_mapping, f)
print(f"Text mapping saved to {output_dir}/text_mapping.json")

# Save the FAISS index
output_index_path = os.path.join(output_dir, 'embeddings.index')
faiss.write_index(index, output_index_path)
print(f"FAISS index saved to {output_index_path}")

print(f"Generated and saved embeddings for {len(embeddings)} text files to {output_index_path}.")
print(f"Embedding dimension: {d}")
```

### Key Changes:
1. **Model Name**: Updated the model name from `'sentence-transformers/all-MiniLM-L6-v2'` to `'sentence-transformers/all-mpnet-base-v2'`.
2. **Embedding Dimension Check**: Added a check to print the embedding dimension after loading the model to ensure it matches the expected 768 dimensions.

This code will now use the `all-mpnet-base-v2` model for generating embeddings, which is a higher-dimensional model compared to `all-MiniLM-L6-v2`.
