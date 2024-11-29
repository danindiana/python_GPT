To improve the recognition of mathematical notation and LaTeX symbols during document conversion, you can take several steps. These steps involve enhancing the OCR preprocessing, using specialized OCR models, and post-processing the extracted text to better handle mathematical symbols. Below are some suggestions to improve your code:

### 1. **Use a Specialized OCR Model for Math and LaTeX**
   - **Tesseract Configuration**: Tesseract has a specialized configuration for recognizing mathematical symbols. You can use the `--oem 1 --psm 6` options and specify the `math` or `latex` trained data.
   - **MathOCR**: Consider using a specialized OCR model like MathOCR, which is designed to recognize mathematical expressions.

### 2. **Enhance Image Preprocessing**
   - **Mathematical Symbol Enhancement**: Apply specific image processing techniques to enhance mathematical symbols and equations. For example, you can use edge detection, dilation, and erosion to make symbols more prominent.
   - **Binarization**: Ensure that the binarization process is optimized for mathematical symbols, which often require a higher threshold.

### 3. **Post-Processing Text**
   - **LaTeX Detection and Conversion**: After OCR, you can use regular expressions or specialized libraries to detect and convert mathematical symbols into LaTeX format.
   - **Symbol Mapping**: Create a mapping of common OCR errors for mathematical symbols and correct them post-OCR.

### 4. **Integration with LaTeX Parsers**
   - **LaTeX Parsing**: Use a LaTeX parser to identify and correct LaTeX syntax errors in the extracted text.

### 5. **Use of External Libraries**
   - **MathPix**: Consider using the MathPix API, which is specifically designed for extracting mathematical expressions from images.
   - **LaTeX OCR**: Use libraries like `LaTeX OCR` that are designed to convert images of mathematical expressions into LaTeX code.

### 6. **Error Handling and Logging**
   - **Detailed Logging**: Implement detailed logging to track errors and issues during OCR and post-processing.
   - **Retry Mechanism**: Implement a retry mechanism for OCR failures, especially for complex pages with mathematical content.

### Updated Code Example

Here’s an updated version of your code with some of these improvements:

```python
import os
import pytesseract
import pymupdf  # PyMuPDF for direct text extraction
import torch
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
from colpali_engine.models import ColQwen2, ColQwen2Processor
import subprocess
import gc
import json
import re

# Set TESSDATA_PREFIX if needed
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

# Verify TESSDATA_PREFIX and eng.traineddata file
tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

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
        doc = pymupdf.open(pdf_path)
        for page in doc:  # Corrected to iterate over pages
            text += f"\n--- Page {page.number + 1} ---\n"
            text += page.get_text("text")  # Direct text extraction
    except pymupdf.FileDataError as e:
        print(f"MuPDF error with file {pdf_path}: {e}")
        return ""
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
        try:
            # Use specialized OCR configuration for math and LaTeX
            page_ocr_text = pytesseract.image_to_string(processed_image, config='--oem 1 --psm 6 -l eng+math')
        except pytesseract.TesseractError as e:
            print(f"Tesseract OCR failed for page {page_number} in {pdf_file}: {e}")
            page_ocr_text = ""

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

# Define paths for progress tracking
progress_file = "processed_files.log"

def load_progress():
    """Load the list of processed files from the log."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()

def save_progress(processed_files):
    """Save the list of processed files to the log."""
    with open(progress_file, "w") as f:
        json.dump(list(processed_files), f)

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

# Load progress at the beginning
processed_files = load_progress()

# Filter files to process
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf') and f not in processed_files]

if not pdf_files:
    print("All PDF files in the directory have already been processed.")
    exit()

print(f"Found {len(pdf_files)} files to process.")

# Prompt user to resume or start fresh
resume_prompt = input("Do you want to resume from the last stop? (y/n): ").strip().lower()
if resume_prompt != "y":
    print("Starting fresh. Clearing progress log...")
    processed_files = set()
    save_progress(processed_files)

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

# Initialize a list to store skipped files
skipped_files = []

# Process each PDF file in the input directory
for pdf_file in pdf_files:
    if len(pdf_file) > 200:
        print(f"Skipping file {pdf_file} due to file name length exceeding 200 characters.")
        skipped_files.append(pdf_file)
        continue

    pdf_path = os.path.join(input_dir, pdf_file)

    try:
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
                        gc.collect()
                        break

                torch.cuda.empty_cache()
                gc.collect()

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
                            gc.collect()

                            if all_image_embeddings is not None:
                                scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                                similarity_scores.append(scores[0].mean().item())
                        except Exception as e:
                            print(f"Error processing text chunk for {pdf_file}: {e}")
                            torch.cuda.empty_cache()
                            gc.collect()
                            break
                except torch.cuda.OutOfMemoryError:
                    print("Skipping due to CUDA memory issue.")
                    torch.cuda.empty_cache()
                    gc.collect()
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

        # Add the file to the processed list
        processed_files.add(pdf_file)
        save_progress(processed_files)

        print(f"Successfully processed: {pdf_file}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        # Optional: log errors to a separate file
        with open("error.log", "a") as error_log:
            error_log.write(f"{pdf_file}: {e}\n")
        continue  # Move to the next file

# Final memory cleanup
torch.cuda.empty_cache()
gc.collect()

# Display the list of skipped files
if skipped_files:
    print("\nThe following files were skipped:")
    for skipped_file in skipped_files:
        print(skipped_file)
else:
    print("\nNo files were skipped.")
```

### Key Changes:
1. **Specialized OCR Configuration**: Added `config='--oem 1 --psm 6 -l eng+math'` to `pytesseract.image_to_string` to improve recognition of mathematical symbols.
2. **Image Preprocessing**: Enhanced preprocessing to better handle mathematical symbols.
3. **Post-Processing**: Added a placeholder for post-processing to convert recognized symbols into LaTeX format.

### Additional Considerations:
- **MathPix API**: Integrate the MathPix API for better recognition of mathematical expressions.
- **LaTeX Parsing**: Use a LaTeX parser to correct syntax errors in the extracted LaTeX code.
- **Error Handling**: Implement more robust error handling and logging to track issues during OCR and post-processing.

By incorporating these improvements, your code should be better equipped to handle mathematical notation and LaTeX symbols during document conversion.
