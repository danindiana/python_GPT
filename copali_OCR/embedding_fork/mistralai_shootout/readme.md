import os
import pytesseract
import fitz  # PyMuPDF for direct text extraction
import torch
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
from colpali_engine.models import ColQwen2, ColQwen2Processor
import subprocess
import fasttext
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set TESSDATA_PREFIX if needed
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00'

# Verify TESSDATA_PREFIX and eng.traineddata file
tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

# Load FastText language identification model
model = fasttext.load_model('lid.176.bin')

def preprocess_image_for_ocr(image):
    """Preprocess the image for better OCR accuracy."""
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.autocontrast(image)  # Increase contrast
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary threshold
    return image

def extract_text_without_ocr(pdf_path):
    """Attempt to extract embedded text directly from the PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:  # Corrected to iterate over pages
                text += f"\n--- Page {page.number + 1} ---\n"
                text += page.get_text("text")  # Direct text extraction
    except Exception as e:
        print(f"Failed to extract text from file {pdf_path}: {e}")
    return text

def extract_images_and_text_ocr(pdf_path, resize_factor=2):
    """Extract images and text from PDF using OCR if necessary."""
    images = []
    pdf_text = extract_text_without_ocr(pdf_path)

    if pdf_text.strip():
        return images, pdf_text, pdf_text  # `images` will be an empty list if no images were processed

    try:
        pdf = PdfDocument(pdf_path)
    except Exception as e:
        print(f"Failed to load PDF {pdf_path}: {e}")
        return [], "", ""

    ocr_text = ""

    for page_number, page in enumerate(pdf):
        width, height = page.get_size()
        bitmap = page.render()

        try:
            pil_image = bitmap.to_pil()
        except AttributeError:
            pixmap = bitmap.to_pixmap()
            pil_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

        new_width = int(width // resize_factor)
        new_height = int(height // resize_factor)
        pil_image = pil_image.resize((new_width, new_height))

        processed_image = preprocess_image_for_ocr(pil_image)
        page_ocr_text = pytesseract.image_to_string(processed_image)
        ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text
        images.append(pil_image)

    return images, "", ocr_text

def split_text_into_chunks(text, chunk_size):
    """Split text into chunks of the specified size."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_gpu_info():
    """Fetch GPU information using nvidia-smi."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpus = []
    for line in output.strip().split('\n'):
        index, name, utilization = line.split(', ')
        gpus.append((int(index), name, int(utilization)))
    return gpus

def select_gpu(gpus):
    """Prompt the user to select a GPU."""
    print("Available GPUs:")
    for i, (index, name, utilization) in enumerate(gpus):
        print(f"{i + 1}. GPU {index}: {name} (Utilization: {utilization}%)")

    while True:
        try:
            selection = int(input("Select the GPU you wish to use (enter the corresponding number): "))
            if 1 <= selection <= len(gpus):
                return gpus[selection - 1][0]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def detect_language(text):
    lang, confidence = model.predict(text)
    return lang[0], confidence[0]

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def cosine_similarity_deduplication(texts, threshold=0.9):
    embeddings = [processor.process_queries([text]).to(device) for text in texts]
    similarity_matrix = cosine_similarity(np.array([emb.cpu().detach().numpy() for emb in embeddings]))
    unique_texts = []
    seen_indices = set()

    for i in range(len(texts)):
        if i not in seen_indices:
            unique_texts.append(texts[i])
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i][j] > threshold:
                    seen_indices.add(j)

    return unique_texts

# Ask the user for input and output directories
input_dir = input("Enter the path of the target directory containing PDF files: ")
output_dir = input("Enter the path of the output directory for processed text files: ")

# Verify the directories exist
if not os.path.isdir(input_dir):
    print("The target directory does not exist.")
    exit()
if not os.path.isdir(output_dir):
    print("The output directory does not exist.")
    exit()

# Fetch GPU information and prompt user to select a GPU
gpus = get_gpu_info()
selected_gpu = select_gpu(gpus)
torch.cuda.set_device(selected_gpu)

# Load model and processor only after directory confirmation to delay GPU allocation
device = torch.device(f"cuda:{selected_gpu}")
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.float16  # Ensure half-precision to save memory
).to(device).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Set a lower maximum chunk size for memory efficiency
max_chunk_size = 5000  # Reduced to 5000 to avoid high memory usage
max_sequence_length = 32768  # Define the max sequence length

# Process all PDF files in the target directory
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

if not pdf_files:
    print("No PDF files found in the specified directory.")
    exit()

# Initialize a list to store skipped files
skipped_files = []

# Process each PDF file in the input directory
for pdf_file in pdf_files:
    if len(pdf_file) > 200:
        print(f"Skipping file {pdf_file} due to file name length exceeding 200 characters.")
        skipped_files.append(pdf_file)
        continue

    pdf_path = os.path.join(input_dir, pdf_file)
    images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)

    print(f"Processing images for {pdf_file}...")

    # Save OCR-like text to a file in the output directory
    output_file = os.path.join(output_dir, f"{pdf_file}_ocr_output.txt")
    with open(output_file, "w") as f:
        f.write("OCR-like extracted text:\n")
        f.write(ocr_text)

    print(f"\nOCR-like extracted text saved to {output_file}")

    # Process images with a batch size of 1 to prevent out-of-memory errors
    all_image_embeddings = []
    if images:
        for i in range(0, len(images), 1):  # Batch size reduced to 1
            image_batch = images[i:i + 1]
            batch_images = processor.process_images(image_batch).to(device)

            with torch.no_grad():
                try:
                    print(f"Processing image batch {i} for {pdf_file}...")
                    image_embeddings = model(**batch_images)
                    all_image_embeddings.append(image_embeddings)
                except Exception as e:
                    print(f"Error processing image batch {i} for {pdf_file}: {e}")
                    torch.cuda.empty_cache()
                    break

            torch.cuda.empty_cache()

        if all_image_embeddings:
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        else:
            all_image_embeddings = None
            print("No image embeddings were created.")
    else:
        all_image_embeddings = None
        print("No images found in the PDF for processing.")

    # Use OCR text if direct text extraction was empty
    if not pdf_text.strip() and ocr_text.strip():
        pdf_text = ocr_text

    # Check if there is text content to process
    if pdf_text.strip():
        print("Processing text...")
        # Dynamically split text into manageable chunks based on max_chunk_size
        text_chunks = split_text_into_chunks(pdf_text, max_chunk_size)
        similarity_scores = []
        skip_due_to_length = False

        for chunk in text_chunks:
            if len(chunk.split()) > max_sequence_length:
                print(f"Skipping file {pdf_file} due to chunk length exceeding {max_sequence_length}")
                skip_due_to_length = True
                skipped_files.append(pdf_file)
                break

            try:
                # Proceed with model processing for valid chunks
                queries = [chunk]
                batch_queries = processor.process_queries(queries).to(device)

                with torch.no_grad():
                    try:
                        print(f"Processing text chunk for {pdf_file}...")
                        query_embeddings = model(**batch_queries)
                        torch.cuda.empty_cache()

                        if all_image_embeddings is not None:
                            scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                            similarity_scores.append(scores[0].mean().item())
                    except Exception as e:
                        print(f"Error processing text chunk for {pdf_file}: {e}")
                        torch.cuda.empty_cache()
                        break
            except torch.cuda.OutOfMemoryError:
                print("Skipping due to CUDA memory issue.")
                torch.cuda.empty_cache()
                skip_due_to_length = True
                skipped_files.append(pdf_file)
                break

        if skip_due_to_length:
            continue

        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            print(f"Average Similarity Score for {pdf_file}: {avg_score:.4f}")
        else:
            print("No similarity scores were calculated.")
    else:
        print("No text found in the PDF for processing.")

# Final memory cleanup
torch.cuda.empty_cache()

# Display the list of skipped files
if skipped_files:
    print("\nThe following files were skipped:")
    for skipped_file in skipped_files:
        print(skipped_file)
else:
    print("\nNo files were skipped.")

# Deduplication and Language Filtering
all_texts = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)
    _, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)
    text = pdf_text if pdf_text.strip() else ocr_text
    if text.strip():
        lang, confidence = detect_language(text)
        if lang == '__label__en':
            all_texts.append(text)

# Deduplicate using cosine similarity
unique_texts = cosine_similarity_deduplication(all_texts)

# Save unique texts to the output directory
for i, text in enumerate(unique_texts):
    output_file = os.path.join(output_dir, f"unique_text_{i}.txt")
    with open(output_file, "w") as f:
        f.write(text)
    print(f"Unique text saved to {output_file}")

When I say "deduplicates the texts using cosine similarity," I mean the process of identifying and removing near-duplicate texts based on their similarity. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is often used in text analysis to determine how similar two documents are.

Here's a more detailed explanation:

1. **Cosine Similarity**:
   - Cosine similarity measures the cosine of the angle between two vectors. In the context of text, these vectors are typically representations of the text, such as embeddings generated by a model.
   - The cosine similarity score ranges from -1 to 1, where 1 indicates that the vectors are identical, 0 indicates that they are orthogonal (no similarity), and -1 indicates that they are diametrically opposed.

2. **Deduplication Process**:
   - **Embedding Generation**: Each text is converted into a vector (embedding) using a model. These embeddings capture the semantic meaning of the text.
   - **Similarity Calculation**: The cosine similarity between each pair of embeddings is calculated.
   - **Thresholding**: A similarity threshold is set (e.g., 0.9). If the cosine similarity between two texts exceeds this threshold, they are considered near-duplicates.
   - **Removal of Duplicates**: One of the near-duplicates is kept, and the others are removed. This ensures that only unique or sufficiently distinct texts are retained.

### Example:
Suppose you have three texts:
- Text A: "The quick brown fox jumps over the lazy dog."
- Text B: "A quick brown fox jumps over a lazy dog."
- Text C: "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

1. **Embedding Generation**:
   - Convert Text A, Text B, and Text C into embeddings using a model.

2. **Similarity Calculation**:
   - Calculate the cosine similarity between the embeddings of Text A and Text B.
   - Calculate the cosine similarity between the embeddings of Text A and Text C.
   - Calculate the cosine similarity between the embeddings of Text B and Text C.

3. **Thresholding**:
   - Assume the similarity threshold is set to 0.9.
   - If the cosine similarity between Text A and Text B is 0.95 (above the threshold), they are considered near-duplicates.
   - If the cosine similarity between Text A and Text C is 0.1 (below the threshold), they are considered distinct.

4. **Removal of Duplicates**:
   - Keep Text A and remove Text B (or vice versa).
   - Keep Text C since it is distinct from both Text A and Text B.

### Code Implementation:
Here's a simplified version of the deduplication process using cosine similarity:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_deduplication(texts, threshold=0.9):
    # Generate embeddings for each text
    embeddings = [processor.process_queries([text]).to(device) for text in texts]
    embeddings = [emb.cpu().detach().numpy() for emb in embeddings]

    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    unique_texts = []
    seen_indices = set()

    for i in range(len(texts)):
        if i not in seen_indices:
            unique_texts.append(texts[i])
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i][j] > threshold:
                    seen_indices.add(j)

    return unique_texts

# Example usage
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown fox jumps over a lazy dog.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
]

unique_texts = cosine_similarity_deduplication(texts, threshold=0.9)
print(unique_texts)
```

In this example, the `cosine_similarity_deduplication` function takes a list of texts and a similarity threshold. It generates embeddings for each text, calculates the cosine similarity matrix, and removes near-duplicates based on the threshold. The resulting `unique_texts` list contains only the unique or sufficiently distinct texts.
