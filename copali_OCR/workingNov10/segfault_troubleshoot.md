A segmentation fault (segfault) is a serious error that occurs when a program tries to access memory it's not allowed to, often leading to a crash.  Debugging segfaults can be tricky, but here's a breakdown of how to investigate them, especially in your context:

**1. Understand the Context**

* **"No images found..."**: This tells us the segfault happened during the text processing phase, not image processing.
* **"Processing text..."**:  The crash occurred while handling the extracted text from the PDF.
* **Likely Culprit**: The most probable cause is within the `split_text_into_chunks`, `model`, or `processor` operations, especially if a large text chunk is causing memory issues.

**2. Debugging Techniques**

* **Enable Core Dumps:**  Ensure your system is configured to generate core dump files when segfaults occur. These files contain a snapshot of the program's memory at the time of the crash.
    * On Linux: `ulimit -c unlimited` (in your shell or script before running the program)
* **Use a Debugger (gdb):**
    1. **Compile with Debug Symbols:** If possible, recompile your Python code with debug symbols (`-g` flag for GCC). This gives the debugger more information.
    2. **Run in gdb:**  `gdb python` then `run your_script.py` within gdb.
    3. **Reproduce the crash:**  Let the program run until it segfaults.
    4. **Inspect the state:** Use commands like `backtrace` (or `bt`) to see the call stack, `print` to inspect variables, `list` to view the code around the crash, and `info locals` to see local variables.
* **Print Statements:**  Add `print` statements to your code to track the program's execution flow and variable values, especially before the suspected crash point. This can help you narrow down the issue.
* **Memory Profiling:** Tools like Valgrind (specifically Memcheck) can help detect memory errors, such as accessing unallocated memory or memory leaks, which might be contributing to the segfault.

**3. Specific Checks for Your Code**

* **Chunk Size:**  Experiment with smaller `max_chunk_size` values to see if the segfault is related to large text chunks.
* **Model Input:** Carefully examine the input you're feeding to the `ColQwen2` model. Large or improperly formatted input could cause problems.
* **Memory Usage:** Monitor memory usage during execution. If it spikes dramatically before the crash, you might have a memory leak or excessive memory allocation.  Tools like `top` or `htop` can help.
* **Error Handling within Loops:** Add `try-except` blocks within the loops that process text chunks to catch potential exceptions and prevent the entire program from crashing.

**Example using gdb:**

```bash
ulimit -c unlimited
gdb python
(gdb) run your_script.py
(gdb) bt  # After the crash
```

**Important Considerations:**

* **Third-party Libraries:** Segfaults can sometimes originate within third-party libraries (like PyMuPDF, pytesseract, or the ColQwen2 model).  Check for updates or known issues with these libraries.
* **System Resources:** Ensure your system has enough available memory and resources to handle the processing.

By carefully analyzing the code, using debugging tools, and systematically investigating potential causes, you can track down the root of the segmentation fault and make your code more stable.

The "Segmentation fault (core dumped)" error is a critical error that indicates the program tried to access a memory location that it was not allowed to access. This can happen due to various reasons, such as:

1. **Memory Corruption**: Errors in the code that lead to memory corruption.
2. **Library Issues**: Problems with the libraries being used (e.g., Tesseract, PyMuPDF, PyTorch).
3. **Hardware Issues**: Rarely, hardware issues can cause segmentation faults.

To diagnose and potentially fix this issue, you can take the following steps:

### 1. **Check for Known Issues with Libraries**
Ensure that all the libraries you are using (Tesseract, PyMuPDF, PyTorch, etc.) are up to date and compatible with each other. Sometimes, updating libraries can resolve such issues.

### 2. **Run the Code in a Debugger**
Running the code in a debugger can help identify the exact line where the segmentation fault occurs. This can provide more insight into what might be causing the issue.

### 3. **Add Debugging Statements**
Add print statements or logging to trace the execution flow and identify where the issue might be occurring.

### 4. **Check for Memory Leaks**
Ensure that there are no memory leaks in your code. Libraries like `valgrind` can help identify memory issues.

### 5. **Simplify the Code**
Try to isolate the problem by simplifying the code. For example, comment out parts of the code to see if the issue persists.

### Updated Code with Debugging Statements

Here is the updated code with additional debugging statements:

```python
import os
import logging
import pytesseract
import fitz  # PyMuPDF for direct text extraction
from PIL import Image, ImageOps
from pypdfium2 import PdfDocument
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor

# Set up logging
logging.basicConfig(filename='ocr_errors.log', level=logging.ERROR)

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Enable dynamic memory segments
os.environ["TESSDATA_PREFIX"] = "/usr/local/share/"

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
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text("text")  # Direct text extraction
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
        logging.error(f"Failed to load PDF document {pdf_path}: {e}")
        return images, "", "Failed to load PDF document."

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
            ocr_text += f"\n--- Page {page_number + 1} ---\n" + page_ocr_text
            images.append(pil_image)
        except pytesseract.pytesseract.TesseractError as e:
            logging.error(f"Tesseract Error on page {page_number + 1} of {pdf_path}: {e}")
            ocr_text += f"\n--- Page {page_number + 1} ---\n" + "OCR failed for this page."
            images.append(pil_image)  # Still append the image for further processing

    return images, "", ocr_text

def split_text_into_chunks(text, chunk_size):
    """Split text into chunks of the specified size."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

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

# Load model and processor only after directory confirmation to delay GPU allocation
device = torch.device("cuda:0")
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
    pdf_path = os.path.join(input_dir, pdf_file)
    try:
        images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)
    except Exception as e:
        logging.error(f"Failed to process PDF file {pdf_file}: {e}")
        skipped_files.append(pdf_file)
        continue

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
                    query_embeddings = model(**batch_queries)
                    torch.cuda.empty_cache()

                    if all_image_embeddings is not None:
                        scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                        similarity_scores.append(scores[0].mean().item())
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
    print("\nThe following files were skipped due to errors:")
    for skipped_file in skipped_files:
        print(skipped_file)
else:
    print("\nNo files were skipped.")
```

### Debugging Steps

1. **Run the Code in a Debugger**:
   - Use a debugger like `pdb` or an IDE with debugging capabilities to step through the code and identify where the segmentation fault occurs.

2. **Check for Memory Leaks**:
   - Use tools like `valgrind` to check for memory leaks and other memory-related issues.

3. **Update Libraries**:
   - Ensure all libraries (Tesseract, PyMuPDF, PyTorch, etc.) are up to date and compatible with each other.

4. **Simplify the Code**:
   - Comment out parts of the code to isolate the issue. For example, try running the code without the image processing part to see if the issue persists.

By following these steps, you should be able to identify and resolve the segmentation fault issue.
