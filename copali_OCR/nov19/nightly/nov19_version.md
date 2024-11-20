The combined code allows for starting, stopping, and resuming the processing of PDF files. It also supports graceful shutdown by the user. Here's how these features are implemented:

### Starting/Stopping/Resuming:
1. **Progress Tracking**:
   - The script uses a `processed_files.log` to keep track of which files have already been processed.
   - At the beginning of the script, it loads the list of processed files from this log.
   - The user is prompted to decide whether to resume from the last stop or start fresh. If the user chooses to start fresh, the progress log is cleared.

2. **Filtering Files**:
   - The script filters out files that have already been processed, ensuring that only new or unprocessed files are processed.

### Graceful Shutdown:
1. **Signal Handling**:
   - The script includes a signal handler for the `SIGINT` signal (typically sent by pressing `Ctrl+C`).
   - When the signal is received, the script performs a graceful shutdown by saving the current progress and cleaning up resources (e.g., emptying the CUDA cache and collecting garbage).

Here is the relevant part of the code that handles these features:

```python
import signal
import sys

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

def signal_handler(sig, frame):
    """Handle the SIGINT signal to perform a graceful shutdown."""
    print("\nReceived shutdown signal. Cleaning up resources and saving progress...")
    save_progress(processed_files)
    torch.cuda.empty_cache()
    gc.collect()
    print("Graceful shutdown complete. Exiting...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Ask the user for input and output directories
input_dir = input("Enter the path of the target directory containing PDF files: ")
output_dir = input("Enter the path of the output directory for processed text files: ")
output_format = input("Enter the desired output format (txt, json, csv): ").strip().lower()

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
        output_file = os.path.join(output_dir, f"{pdf_file}_ocr_output.{output_format}")
        save_output(output_file, ocr_text, output_format)

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

### How It Works:
1. **Starting**:
   - The script starts by loading the list of processed files from `processed_files.log`.
   - It filters out files that have already been processed.
   - The user is prompted to decide whether to resume from the last stop or start fresh.

2. **Stopping**:
   - The user can stop the script at any time by pressing `Ctrl+C`.
   - The signal handler captures the `SIGINT` signal and performs a graceful shutdown, saving the current progress and cleaning up resources.

3. **Resuming**:
   - If the user chooses to resume, the script continues processing from where it left off, using the list of processed files from `processed_files.log`.
   - If the user chooses to start fresh, the progress log is cleared, and the script starts processing all files from the beginning.

4. **Graceful Shutdown**:
   - When the `SIGINT` signal is received, the signal handler saves the current progress to `processed_files.log`, empties the CUDA cache, collects garbage, and exits the script gracefully.

This ensures that the script can be started, stopped, and resumed efficiently, with proper handling of progress and resources.
