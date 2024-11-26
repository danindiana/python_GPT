from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import glob

# Ensure at least 2 GPUs are available
if torch.cuda.device_count() < 2:
    raise RuntimeError("Multi-GPU processing requires at least 2 GPUs. Only 1 GPU is detected.")

# Define model name
model_name = "MendelAI/nv-embed-v2-ontada-twab-peft"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer loaded successfully.")

# Load the SentenceTransformer model with device mapping
print("Loading model and distributing across GPUs...")
model = SentenceTransformer(
    model_name,
    trust_remote_code=True
)

# Confirm GPU utilization
print(f"Model successfully loaded. Available GPUs: {torch.cuda.device_count()}")

# Function to process sentences
def process_sentences(sentences):
    """Encode sentences in batches."""
    print(f"Processing {len(sentences)} sentences...")
    embeddings = model.encode(sentences, batch_size=2, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

# Example: Process sentences from input files
target_directory = input("Enter the target directory containing .txt or .json files: ")
file_paths = glob.glob(f"{target_directory}/*.txt") + glob.glob(f"{target_directory}/*.json")

if not file_paths:
    raise FileNotFoundError(f"No .txt or .json files found in directory: {target_directory}")

# Process files
all_embeddings = []
for file_path in file_paths:
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        sentences = content.split("\n")  # Assuming each line is a sentence
        embeddings = process_sentences(sentences)
        all_embeddings.append((file_path, embeddings))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Summary
print(f"Processed {len(all_embeddings)} files successfully.")
