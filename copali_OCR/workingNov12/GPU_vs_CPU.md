To determine which elements of the program are run on the GPU and which are run on the CPU, we need to identify the parts of the code that involve GPU operations and those that do not. Here's a breakdown of the key elements in your program:

### Elements Run on the GPU:
1. **Model Loading and Evaluation**:
   - The model (`ColQwen2`) is loaded onto the GPU using `.to(device)`.
   - The model is evaluated in inference mode using `.eval()`.

   ```python
   device = torch.device("cuda:0")
   model = ColQwen2.from_pretrained(
       "vidore/colqwen2-v0.1",
       torch_dtype=torch.float16  # Ensure half-precision to save memory
   ).to(device).eval()
   ```

2. **Image Processing**:
   - The images are processed and moved to the GPU using `.to(device)`.

   ```python
   batch_images = processor.process_images(image_batch).to(device)
   ```

3. **Text Processing**:
   - The text chunks are processed and moved to the GPU using `.to(device)`.

   ```python
   batch_queries = processor.process_queries(queries).to(device)
   ```

4. **Model Inference**:
   - The model performs inference on the GPU.

   ```python
   with torch.no_grad():
       image_embeddings = model(**batch_images)
       query_embeddings = model(**batch_queries)
   ```

5. **Memory Management**:
   - GPU memory is freed using `torch.cuda.empty_cache()`.

   ```python
   torch.cuda.empty_cache()
   ```

### Elements Run on the CPU:
1. **File Handling and I/O Operations**:
   - Reading and writing files.
   - Directory verification.

   ```python
   input_dir = input("Enter the path of the target directory containing PDF files: ")
   output_dir = input("Enter the path of the output directory for processed text files: ")

   if not os.path.isdir(input_dir):
       print("The target directory does not exist.")
       exit()
   if not os.path.isdir(output_dir):
       print("The output directory does not exist.")
       exit()
   ```

2. **Image Preprocessing**:
   - Converting images to grayscale.
   - Applying contrast and thresholding.

   ```python
   def preprocess_image_for_ocr(image):
       image = image.convert("L")  # Convert to grayscale
       image = ImageOps.autocontrast(image)  # Increase contrast
       image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Apply binary threshold
       return image
   ```

3. **Text Extraction and Processing**:
   - Extracting text from PDFs.
   - Splitting text into chunks.

   ```python
   def extract_text_without_ocr(pdf_path):
       text = ""
       try:
           with fitz.open(pdf_path) as doc:
               for page_num in range(len(doc)):
                   page = doc[page_num]
                   text += f"\n--- Page {page_num + 1} ---\n"
                   text += page.get_text("text")  # Direct text extraction
       except fitz.FileDataError:
           print(f"Failed to open file {pdf_path}. Skipping...")
       return text

   def split_text_into_chunks(text, chunk_size):
       words = text.split()
       return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
   ```

4. **OCR Operations**:
   - Performing OCR on preprocessed images.

   ```python
   page_ocr_text = pytesseract.image_to_string(processed_image)
   ```

5. **Debugging and Logging**:
   - Printing debug information.

   ```python
   print(f"Processing image batch {i} for {pdf_file}...")
   print(f"Processing text chunk for {pdf_file}...")
   ```

### Summary:
- **GPU Operations**: Model loading, evaluation, image and text processing, model inference, and memory management.
- **CPU Operations**: File handling, image preprocessing, text extraction, OCR operations, and debugging/logging.

### Optimizing GPU Usage:
To optimize GPU usage, you can:
1. **Reduce Batch Size**: As mentioned earlier, reduce the batch size to minimize GPU memory usage.
2. **Use Mixed Precision**: Ensure that you are using mixed precision (float16) to reduce memory usage.
3. **Free GPU Memory**: Ensure that GPU memory is freed after each batch processing using `torch.cuda.empty_cache()`.
4. **Optimize Data**: Reduce the size of the images or text chunks to minimize memory usage.
5. **Use CPU for Certain Operations**: Move certain operations that do not require GPU to the CPU to free up GPU memory.
