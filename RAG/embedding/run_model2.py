from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import os

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
    model = SentenceTransformer("MendelAI/nv-embed-v2-ontada-twab-peft", trust_remote_code=True)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel("MendelAI/nv-embed-v2-ontada-twab-peft", device_ids=[0, 1])
    print(f"Using {torch.cuda.device_count()} GPUs")
else:
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Run inference
sentences = [
    'Instruct: Given a question, retrieve passages that answer the question. Query: what is the total dose administered in the EBRT Intensity Modulated Radiation Therapy?',
    'Source: SOAP_Note. Date: 2020-03-13. Context: MV electrons.\n \n FIELDS:\n The right orbital mass and right cervical lymph nodes were initially treated with a two arc IMRT plan. Arc 1: 11.4 x 21 cm. Gantry start and stop angles 178 degrees / 182 degrees. Arc 2: 16.4 x 13.0 cm. Gantry start ',
    'Source: Radiology. Date: 2023-09-18. Context: : >60\n \n Contrast Type: OMNI 350\n   Volume: 80ML\n \n Lot_: ________\n \n Exp. date: 05/26 \n Study Completed: CT CHEST W\n \n Reading Group:BCH \n  \n   Prior Studies for Comparison: 06/14/23 CT CHEST W RMCC  \n \n ________ ______\n  ',
]

try:
    embeddings = model.module.encode(sentences) if isinstance(model, torch.nn.DataParallel) else model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
except Exception as e:
    print(f"Error encoding sentences: {e}")

# Get the similarity scores for the embeddings
try:
    similarities = model.module.cos_sim(embeddings, embeddings) if isinstance(model, torch.nn.DataParallel) else model.cos_sim(embeddings, embeddings)
    print(f"Similarities shape: {similarities.shape}")
except Exception as e:
    print(f"Error calculating similarities: {e}")