from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import os
import glob
import time

# Clear cache
os.system("rm -rf ~/.cache/huggingface/transformers")

# Ensure the tokenizer is compatible
try:
    tokenizer = AutoTokenizer.from_pretrained("MendelAI/nv-embed-v2-ontada-twab-peft", trust_remote_code=True)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load the model with trust_remote_code=True and move it to the GPU
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("MendelAI/nv-embed-v2-ontada-twab-peft", trust_remote_code=True, device=device)
    print("Model loaded successfully")
    
    # Apply additional model settings
    model.max_seq_length = 2768
    model.tokenizer.padding_side = "right"
    
except Exception as e:
    print(f"Error loading model: {e}")

# Prompt the user for the target directory
target_directory = input("Enter the target directory: ")

# List all .txt and .json files in the target directory
file_paths = glob.glob(os.path.join(target_directory, "*.txt")) + glob.glob(os.path.join(target_directory, "*.json"))

# Initialize file count
file_count = 0

# Process each file
for file_path in file_paths:
    try:
        start_time = time.time()  # Start timing
        print(f"Processing file {file_path}...")
        
        with open(file_path, 'r') as file:
            content = file.read()
            sentences = content.split('\n')  # Assuming each line is a separate sentence
            embeddings = model.encode(sentences, batch_size=1, device=device)  # Reduce batch size and move to GPU
            print(f"Embeddings shape for {file_path}: {embeddings.shape}")
            
            # Get the similarity scores for the embeddings
            similarities = model.similarity(embeddings, embeddings)
            print(f"Similarities shape for {file_path}: {similarities.shape}")
        
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"Finished processing {file_path} in {elapsed_time:.2f} seconds.")
        
        file_count += 1
        print(f"Files processed so far: {file_count}/{len(file_paths)}")
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print(f"Total files processed: {file_count}")