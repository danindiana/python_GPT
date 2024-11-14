import torch
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Initialize the model and processor
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.bfloat16,
).eval()

# Explicitly move the model to the second GPU
device = torch.device("cuda:1")
model.to(device)

processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Create a sample input
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "Is attention really all you need?",
    "What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch_images = processor.process_images(images).to(device)
batch_queries = processor.process_queries(queries).to(device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

# Print the shape of the embeddings
print("Image Embeddings Shape:", image_embeddings.shape)
print("Query Embeddings Shape:", query_embeddings.shape)
