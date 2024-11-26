from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import os
import glob

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

# Load the model with trust_remote_code=True
try:
    model = SentenceTransformer("MendelAI/nv-embed-v2-ontada-twab-peft", trust_remote_code=True, device='cpu')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Prompt the user for the target directory
target_directory = input("Enter the target directory: ")

# List all .txt and .json files in the target directory
file_paths = glob.glob(os.path.join(target_directory, "*.txt")) + glob.glob(os.path.join(target_directory, "*.json"))

# Process each file
for file_path in file_paths:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            sentences = content.split('\n')  # Assuming each line is a separate sentence
            embeddings = model.encode(sentences, batch_size=1)  # Reduce batch size
            print(f"Embeddings shape for {file_path}: {embeddings.shape}")
            
            # Get the similarity scores for the embeddings
            similarities = model.similarity(embeddings, embeddings)
            print(f"Similarities shape for {file_path}: {similarities.shape}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")