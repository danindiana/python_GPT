Given that the `eng.traineddata` file exists, Tesseract should be functional. However, adding robust error handling and crash recovery will make your `pdf_ocr_gpu_selectionv2.py` script more resilient. Here's how you can modify the script to pick up where it left off in case of a crash:

---

### **Crash Recovery and Resumption**
1. **Track Progress**:
   - Maintain a log of processed files in a progress file (e.g., `processed_files.log`).
   - Each time a file is successfully processed, append its name to this log.

2. **Detect Unprocessed Files**:
   - At the start of the script, check the log to identify which files have already been processed.
   - Skip those files during subsequent runs.

3. **Prompt for Resumption**:
   - Allow the user to choose whether to resume processing from the last stopped point.

---

### **Implementation**:

Below is a suggested modification for `pdf_ocr_gpu_selectionv2.py`:

#### **Add Progress Logging**
```python
import json
import os

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
```

#### **Filter Files to Process**
```python
# Load progress at the beginning
processed_files = load_progress()

# Filter files to process
pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf') and f not in processed_files]

if not pdf_files:
    print("All PDF files in the directory have already been processed.")
    exit()

print(f"Found {len(pdf_files)} files to process.")
```

#### **Prompt for Resumption**
```python
# Prompt user to resume or start fresh
resume_prompt = input("Do you want to resume from the last stop? (y/n): ").strip().lower()
if resume_prompt != "y":
    print("Starting fresh. Clearing progress log...")
    processed_files = set()
    save_progress(processed_files)
```

#### **Update Progress During Processing**
```python
# Process each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)

    try:
        # Process the file (as in your existing logic)
        images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)
        # Add additional processing logic...

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
```

---

### **User Experience**
1. When the script starts:
   - It checks for a `processed_files.log` file.
   - If it exists, it asks the user if they want to resume or start fresh.

2. If the script crashes or halts:
   - Upon restarting, the script identifies unprocessed files and resumes from the last stopped point.

3. Skipped or errored files:
   - Errors are logged in an `error.log` file for review.

---

### **Testing Steps**
- Ensure you test the updated script with a mix of successful and failed PDF processing.
- Interrupt the script manually (e.g., via `Ctrl+C`) to confirm it resumes as expected.

Let me know if you need further clarification or assistance in integrating this!
