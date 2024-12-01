import os
import subprocess
import pytesseract
import fitz  # PyMuPDF for direct text extraction
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
import pikepdf
from PyPDF2 import PdfReader

# Set TESSDATA_PREFIX if needed
os.environ["TESSDATA_PREFIX"] = "/usr/local/share/"

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
            num_pages = len(doc)
            print(f"Number of pages: {num_pages}")
            for page_num in range(num_pages):
                print(f"Processing page {page_num + 1}")
                try:
                    page = doc[page_num]
                    page_text = page.get_text("text")
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                except RuntimeError as e:
                    print(f"Error processing page {page_num + 1}: {e}")
                    continue
    except RuntimeError as e:
        print(f"Error opening file {pdf_path}: {e}")
        return ""
    return text

def extract_images_and_text_ocr(pdf_path, resize_factor=2):
    """Extract images and text from PDF using OCR if necessary."""
    images = []
    pdf_text = extract_text_without_ocr(pdf_path)

    if pdf_text.strip():
        return images, pdf_text, pdf_text  # `images` will be an empty list if no images were processed

    pdf = PdfDocument(pdf_path)
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

def repair_pdf(input_path, output_path):
    try:
        with pikepdf.open(input_path) as pdf:
            pdf.save(output_path)
        print(f"Repaired PDF saved as: {output_path}")
        return output_path
    except pikepdf.PdfError as e:
        print(f"Failed to repair PDF: {e}")
        return None

def validate_pdf(pdf_path):
    try:
        PdfReader(pdf_path)
        return True
    except Exception as e:
        print(f"Invalid PDF {pdf_path}: {e}")
        return False

def process_pdf_with_fallback(pdf_path):
    try:
        pdf_text = extract_text_without_ocr(pdf_path)
        if not pdf_text.strip():
            print(f"No text extracted from {pdf_path}, falling back to OCR.")
            _, _, ocr_text = extract_images_and_text_ocr(pdf_path)
            return ocr_text
        return pdf_text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def safe_extract(pdf_path):
    try:
        return process_pdf_with_fallback(pdf_path)
    except Exception as e:
        print(f"Unhandled error with {pdf_path}: {e}")
        return ""

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
selected_physical_index = select_gpu(gpus)

# Set CUDA_VISIBLE_DEVICES to the selected physical GPU index
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_physical_index)

# Verify device setup
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"Available devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Set the device to "cuda:0"
device = torch.device("cuda:0")

# Load the model and processor
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.float16,
).to(device).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Set a lower maximum chunk size for memory efficiency
max_chunk_size = 2000  # Adjusted chunk size to prevent exceeding token limits

# Process all PDF files in the target directory
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

if not pdf_files:
    print("No PDF files found in the specified directory.")
    exit()

# Process each PDF file in the input directory
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)

    # Repair the PDF if needed
    repaired_pdf_path = repair_pdf(pdf_path, pdf_path.replace(".pdf", "_repaired.pdf"))
    if repaired_pdf_path:
        pdf_path = repaired_pdf_path

    # Validate the PDF
    if not validate_pdf(pdf_path):
        print(f"Skipping {pdf_path} due to validation issues.")
        continue

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
                image_embeddings = model(**batch_images)
                all_image_embeddings.append(image_embeddings)

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
        words = pdf_text.split()
        text_chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

        # Further split chunks that exceed the model's token limit
        tokenizer = processor.tokenizer  # Assuming the processor has a tokenizer attribute
        max_token_limit = 32768  # Model's maximum sequence length

        final_chunks = []
        for chunk in text_chunks:
            tokens = tokenizer.encode(chunk)
            if len(tokens) > max_token_limit:
                # Split the chunk into smaller parts
                sub_chunks = []
                current_sub_chunk = []
                current_token_count = 0
                for token in tokens:
                    if current_token_count + len(token) > max_token_limit:
                        # Encode the current sub-chunk
                        sub_chunks.append(tokenizer.decode(current_sub_chunk))
                        current_sub_chunk = [token]
                        current_token_count = len(token)
                    else:
                        current_sub_chunk.append(token)
                        current_token_count += len(token)
                # Add any remaining sub-chunk
                if current_sub_chunk:
                    sub_chunks.append(tokenizer.decode(current_sub_chunk))
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        similarity_scores = []

        for chunk in final_chunks:
            # Truncate chunk to ensure it does not exceed the maximum sequence length
            if len(tokenizer.encode(chunk)) > max_token_limit:
                chunk = tokenizer.decode(tokenizer.encode(chunk)[:max_token_limit])

            queries = [chunk]
            batch_queries = processor.process_queries(queries).to(device)

            with torch.no_grad():
                query_embeddings = model(**batch_queries)
                torch.cuda.empty_cache()  # Free up memory after each embedding pass

                # Calculate similarity scores for each chunk
                if all_image_embeddings is not None:
                    scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                    similarity_scores.append(scores[0].mean().item())

        # Calculate average similarity score across chunks
        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            print(f"Average Similarity Score for {pdf_file}: {avg_score:.4f}")
        else:
            print("No similarity scores were calculated.")
    else:
        print("No text found in the PDF for processing.")

# Final memory cleanup
torch.cuda.empty_cache()
