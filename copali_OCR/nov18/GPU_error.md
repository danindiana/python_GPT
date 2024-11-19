Your script is encountering several issues, notably with Tesseract configuration and potential memory management problems. Let's address them systematically:

---

### **Key Issues and Fixes:**

1. **Tesseract Configuration Issue:**
   - The error indicates that Tesseract cannot find the `eng.traineddata` file.
   - Ensure that the `TESSDATA_PREFIX` environment variable is correctly pointing to the directory containing `tessdata`. Double-check that:
     ```bash
     ls /usr/share/tesseract-ocr/5/tessdata/eng.traineddata
     ```
     This should confirm the presence of the `eng.traineddata` file.

   - If missing, install the required language data with:
     ```bash
     sudo apt-get install tesseract-ocr-eng
     ```
     or reinstall Tesseract completely:
     ```bash
     sudo apt-get install tesseract-ocr
     ```

2. **MuPDF Error with XREF Size:**
   - This suggests a corrupted or unsupported PDF structure.
   - Add error handling for specific problematic PDFs in the MuPDF processing step:
     ```python
     try:
         with fitz.open(pdf_path) as doc:
             # process document
     except fitz.fitz.FileDataError as e:
         print(f"MuPDF error with file {pdf_path}: {e}")
         skipped_files.append(pdf_file)
         continue
     ```

   - You might also consider using alternative libraries, like PyPDF2 or PDFPlumber, for text extraction.

3. **Pytesseract Processing Issue:**
   - The `pytesseract.image_to_string` error suggests an issue with preprocessing or Tesseract not being initialized correctly.
   - Add a fallback mechanism if Tesseract fails:
     ```python
     try:
         page_ocr_text = pytesseract.image_to_string(processed_image)
     except pytesseract.TesseractError as e:
         print(f"Tesseract OCR failed for page {page_number} in {pdf_file}: {e}")
         page_ocr_text = ""
     ```

4. **Memory Management with GPUs:**
   - CUDA out-of-memory issues are often caused by allocating too much to the GPU at once.
   - Use smaller batches and enforce periodic garbage collection:
     ```python
     import gc
     torch.cuda.empty_cache()
     gc.collect()
     ```
   - Ensure half-precision (`torch.float16`) is consistently applied, and adjust `max_chunk_size` and `resize_factor` as needed.

5. **Skipped Files Logging:**
   - Some files are being skipped without clear reasons or logged details.
   - Add verbose logging to keep track of skipped files:
     ```python
     with open("skipped_files.log", "a") as log_file:
         log_file.write(f"{pdf_file}: Reason for skipping\n")
     ```

---

### **Refactored Steps for Robustness:**

- **Tesseract Setup Check:**
  At the start of your script, validate Tesseract setup and data file paths:
  ```python
  def check_tesseract_setup():
      if not os.path.exists(tessdata_path):
          raise FileNotFoundError(f"TESSDATA_PREFIX is not correctly set: {tessdata_path}")
      if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
          raise FileNotFoundError("eng.traineddata is missing. Reinstall Tesseract or its language data.")
  ```

- **Better Error Handling:**
  Wrap critical processing steps with robust exception handling to log and continue processing the next file.

- **Testing with Problematic Files:**
  Run isolated tests on the problematic file (`mdb-paper.pdf`) with reduced complexity to isolate the exact error:
  ```python
  python3 -m pdb your_script.py
  ```

- **Output Skipped Files:**
  Log all skipped files in a structured format (e.g., JSON or CSV) for review.

---

Would you like help rewriting specific sections, running tests, or debugging further?



import os
import pytesseract
import fitz  # PyMuPDF for direct text extraction
import torch
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
from colpali_engine.models import ColQwen2, ColQwen2Processor
import subprocess

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
        return images, pdf_text, pdf_text  # images will be an empty list if no images were processed

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


Deepseek error analysis:

The error message you're encountering indicates that the Tesseract OCR engine is unable to find the `eng.traineddata` file, which is necessary for recognizing English text. This is likely due to the `TESSDATA_PREFIX` environment variable not being set correctly or the file not being present in the specified directory.

Additionally, the MuPDF error suggests that there is a problem with the PDF file itself, which might be causing issues during text extraction.

Here are some steps to troubleshoot and resolve these issues:

### 1. Verify Tesseract Installation and `TESSDATA_PREFIX`
Ensure that Tesseract is installed correctly and that the `TESSDATA_PREFIX` environment variable is set to the correct path.

#### Check Tesseract Installation
```bash
tesseract --version
```

#### Verify `TESSDATA_PREFIX`
Make sure the `TESSDATA_PREFIX` environment variable is set correctly. You can do this in your script or in your shell configuration file (e.g., `.bashrc`, `.zshrc`).

```python
import os
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'
```

#### Verify `eng.traineddata` Exists
Check if the `eng.traineddata` file exists in the specified directory:

```python
tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
if not os.path.exists(tessdata_path):
    raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")
```

### 2. Handle PDF Errors Gracefully
The MuPDF error suggests that there is a problem with the PDF file. You can add error handling to skip problematic files and continue processing the rest.

```python
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
            page_ocr_text = pytesseract.image_to_string(processed_image)
        except pytesseract.pytesseract.TesseractError as e:
            print(f"Tesseract error on page {page_number + 1} of {pdf_path}: {e}")
            continue

        ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text
        images.append(pil_image)

    return images, "", ocr_text
```

### 3. Handle Memory Issues
Ensure that you are handling memory issues gracefully, especially when dealing with large PDFs or images. You can add more robust error handling and memory management.

### 4. Final Cleanup
Ensure that you are cleaning up resources properly, especially when dealing with GPU memory.

```python
# Final memory cleanup
torch.cuda.empty_cache()
```

### Summary
- Verify Tesseract installation and `TESSDATA_PREFIX`.
- Handle PDF errors gracefully.
- Add robust error handling and memory management.
- Ensure proper resource cleanup.

By following these steps, you should be able to resolve the issues and improve the robustness of your script.
