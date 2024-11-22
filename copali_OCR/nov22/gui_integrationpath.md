To further enhance your PDF processing pipeline, here's an example of how you can implement a simple GUI using Python and Tkinter. This GUI will allow users to select input and output directories, choose the desired output format, and enter the path to their mapping configuration file.

The main benefits of using a GUI are:

1. **User-friendliness**: Users can interact with your application through a graphical interface without needing to understand or type command-line arguments.
2. **Visual feedback**: The GUI can provide real-time updates on the progress and status of the processing tasks.
3. **Easier configuration management**: Users can easily modify input parameters without having to edit scripts directly.

Here's an example implementation of such a GUI using Tkinter:

```python
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class PDFProcessingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF Text Remapper")
        self.root.geometry("400x300")

        # Variables to store user inputs
        self.input_dir = None
        self.output_dir = None
        self.output_format = None
        self.mapping_config_path = None

        # Create and grid GUI elements
        tk.Label(self.root, text="Input Directory:").grid(row=0, column=0, padx=10, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_input_dir).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Output Directory:").grid(row=1, column=0, padx=10, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_output_dir).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Output Format (txt/csv/json):").grid(row=2, column=0, padx=10, pady=5)
        tk.Entry(self.root, width=10, textvariable=self.output_format).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.root, text="Mapping Configuration File:").grid(row=3, column=0, padx=10, pady=5)
        tk.Button(self.root, text="Browse...", command=self.browse_mapping_config).grid(row=3, column=1, padx=5, pady=5)

        tk.Button(self.root, text="Start Processing", command=self.start_processing).grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def browse_input_dir(self):
        self.input_dir = filedialog.askdirectory()
        if not self.input_dir:
            messagebox.showwarning("Warning", "No input directory selected.")
        else:
            print(f"Selected Input Directory: {self.input_dir}")

    def browse_output_dir(self):
        self.output_dir = filedialog.askdirectory()
        if not self.output_dir:
            messagebox.showwarning("Warning", "No output directory selected.")
        else:
            print(f"Selected Output Directory: {self.output_dir}")

    def browse_mapping_config(self):
        self.mapping_config_path = filedialog.askopenfilename(initialdir=".", title="Select file",
                                                              filetypes=(("JSON files", "*.json"), ("all files", "*.*")))
        if not self.mapping_config_path:
            messagebox.showwarning("Warning", "No mapping configuration file selected.")
        else:
            print(f"Selected Mapping Configuration File: {self.mapping_config_path}")

    def start_processing(self):
        # Validate user inputs
        if not all([self.input_dir, self.output_dir, self.output_format.get(), self.mapping_config_path]):
            messagebox.showwarning("Warning", "Please fill in all fields.")
            return

        # Perform processing tasks here using the input variables
        print(f"Processing PDF files in '{self.input_dir}' with output format '{self.output_format.get()}' and mapping configuration '{self.mapping_config_path}'")

        # Close the GUI window after starting the processing task
        self.root.destroy()

if __name__ == "__main__":
    app = PDFProcessingGUI()
    app.root.mainloop()
```

To integrate this GUI with your existing PDF processing pipeline, you can modify `start_processing` method to call your main script with the input variables gathered from the GUI. Here's an example of how to do that:

```python
import subprocess

def start_processing(self):
    # Validate user inputs
    if not all([self.input_dir, self.output_dir, self.output_format.get(), self.mapping_config_path]):
        messagebox.showwarning("Warning", "Please fill in all fields.")
        return

    # Construct the command to run your main script with GUI-provided arguments
    cmd = ["python", "main_script.py", self.input_dir, self.output_dir, self.output_format.get(), self.mapping_config_path]

    # Run the command using subprocess
    try:
        subprocess.run(cmd)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while running the processing task:\n{str(e)}")

    # Close the GUI window after starting the processing task
    self.root.destroy()
```

This approach allows you to keep your main PDF processing script separate from the GUI, making it easier to maintain and update each component independently.

Based on the provided file structure, here's an updated plan to integrate the re-mapping functionality and GUI into your existing project:

1. **Create a new module for re-mapping (`text_remapper.py`)**:
   - Define the `remap_text` function in this module as discussed earlier.

2. **Update `main_scriptv4JSONL.py`** to include the following changes:
   - Import and use the `remap_text` function from `text_remapper.py`.
   - Add a new argument to accept the mapping configuration file path.
   - Load the mapping configuration JSON file using the provided path.
   - Call the `remap_text` function after extracting text, passing in the combined OCR text and loaded mapping configuration.
   - Save the re-mapped output in the desired format (JSON, CSV, or TXT).
   - Update the progress tracker to handle re-mapped files.

Here's a sample structure for your updated `main_scriptv4JSONL.py` file:

```python
import os
import argparse
import json
import csv
from preprocessing import preprocess_image_for_ocr, split_text_into_chunks
from pdf_ocr_utils import extract_images_and_text_ocr
from gpu_selection import get_gpu_info, select_gpu
from model_utils import load_model_and_processor
from progress_tracker import load_progress, save_progress
from error_logger import log_error
from text_remapper import remap_text

def main(input_dir, output_dir, output_format, mapping_config_path):
    # Set TESSDATA_PREFIX if needed
    os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

    # Verify TESSDATA_PREFIX and eng.traineddata file
    tessdata_path = os.path.join(os.environ["TESSDATA_PREFIX"], "tessdata")
    if not os.path.exists(tessdata_path):
        raise FileNotFoundError(f"The directory {tessdata_path} does not exist. Please set TESSDATA_PREFIX correctly.")
    if not os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
        raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")

    # Load mapping configuration
    with open(mapping_config_path, "r") as f:
        mapping_config = json.load(f)

    # Verify the directories exist
    if not os.path.isdir(input_dir):
        print("The target directory does not exist.")
        return
    if not os.path.isdir(output_dir):
        print("The output directory does not exist.")
        return

    # Load progress at the beginning
    processed_files = load_progress()

    # Filter files to process
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf') and f not in processed_files]

    if not pdf_files:
        print("All PDF files in the directory have already been processed.")
        return

    print(f"Found {len(pdf_files)} files to process.")

    # Fetch GPU information and prompt user to select a GPU
    gpus = get_gpu_info()
    selected_gpu = select_gpu(gpus)
    torch.cuda.set_device(selected_gpu)

    # Load model and processor only after directory confirmation to delay GPU allocation
    device = torch.device(f"cuda:{selected_gpu}")
    model, processor = load_model_and_processor(device)

    # Initialize a list to store skipped files
    skipped_files = []

    # Process each PDF file in the input directory
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        try:
            images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path)

            print(f"Processing file {pdf_file}...")

            # Combine extracted text for output
            combined_text = ocr_text if not pdf_text.strip() else pdf_text

            if not combined_text.strip():
                print(f"No text found in {pdf_file}. Skipping...")
                skipped_files.append(pdf_file)
                continue

            # Apply re-mapping after extracting OCR text
            re_mapped_text = remap_text(combined_text, mapping_config)

            # Save output in the selected format
            output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_remapped_output.{output_format}")
            if output_format == 'txt':
                # Combine all text into a single string
                combined_text_with_header = f"Re-mapped OCR-like extracted text:\n{json.dumps(re_mapped_text, indent=4)}\n"

                # Write the content in a single operation
                with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
                    f.write(combined_text_with_header)
            elif output_format == 'csv':
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(['Section Name', 'Content'])
                    for section, lines in re_mapped_text.items():
                        for line in lines:
                            writer.writerow([section, line])
            elif output_format == 'json':
                with open(output_file, "w") as f:
                    json.dump(re_mapped_text, f, indent=4)

            print(f"Successfully processed: {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            log_error(pdf_file, str(e))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Text Remapper")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to the input directory containing PDF files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory for processed text files.")
    parser.add_argument("-f", "--output_format", choices=['txt', 'csv', 'json'], default='json', help="Desired output format (default: json)")
    parser.add_argument("-m", "--mapping_config_path", required=True, help="Path to the mapping configuration file (JSON)")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.output_format, args.mapping_config_path)
```

3. **Create a new GUI module (`gui.py`)**:
   - Implement the Tkinter-based GUI as described in the previous answer.
   - Update the `start_processing` method to call your updated `main_scriptv4JSONL.py` script with the input variables gathered from the GUI.

Here's an example structure for your `gui.py` file:

```python
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

class PDFProcessingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF Text Remapper")
        self.root.geometry("400x300")

        # Variables to store user inputs
        self.input_dir = None
        self.output_dir = None
        self.output_format = tk.StringVar(value='json')
        self.mapping_config_path = None

        # Create and grid GUI elements
        # ... (Same as before)

    def browse_input_dir(self):
        self.input_dir = filedialog.askdirectory()
        if not self.input_dir:
            messagebox.showwarning("Warning", "No input directory selected.")
        else:
            print(f"Selected Input Directory: {self.input_dir}")

    def browse_output_dir(self):
        self.output_dir = filedialog.askdirectory()
        if not self.output_dir:
            messagebox.showwarning("Warning", "No output directory selected.")
        else:
            print(f"Selected Output Directory: {self.output_dir}")

    def browse_mapping_config(self):
        self.mapping_config_path = filedialog.askopenfilename(initialdir=".", title="Select file",
                                                              filetypes=(("JSON files", "*.json"), ("all files", "*.*")))
        if not self.mapping_config_path:
            messagebox.showwarning("Warning", "No mapping configuration file selected.")
        else:
            print(f"Selected Mapping Configuration File: {self.mapping_config_path}")

    def start_processing(self):
        # Validate user inputs
        if not all([self.input_dir, self.output_dir, self.mapping_config_path]):
            messagebox.showwarning("Warning", "Please fill in all fields.")
            return

        # Construct the command to run your main script with GUI-provided arguments
        cmd = ["python", "main_scriptv4JSONL.py", "-i", self.input_dir, "-o", self.output_dir, "-f", self.output_format.get(), "-m", self.mapping_config_path]

        # Run the command using subprocess
        try:
            subprocess.run(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while running the processing task:\n{str(e)}")

        # Close the GUI window after starting the processing task
        self.root.destroy()

if __name__ == "__main__":
    app = PDFProcessingGUI()
    app.root.mainloop()
```

With these modifications, you should have an updated project structure that includes re-mapping functionality and a simple Tkinter-based GUI for user interaction. The GUI will allow users to select input and output directories, choose the desired output format, and enter the path to their mapping configuration file. After clicking "Start Processing," the GUI will run your main script with the provided arguments and display any error messages that occur during processing.

To use this updated project structure, replace your existing `main_scriptv4JSONL.py` and create new files named `text_remapper.py` and `gui.py` according to the provided examples. Make sure to update the file paths in the argparse parser of `main_scriptv4JSONL.py` if needed.

After updating the project structure, you can run the GUI using the command:

```
python gui.py
```

This will launch the PDF Text Remapper GUI, allowing users to interact with your processing pipeline through a graphical interface.

Here's a flowchart and pseudocode representation of the updated project structure and workflow, organized in a tree-like directory format:

**Project Structure:**

```
pdf_text_remapper/
│
├── combined_output.json
├── combined_output.py
├── combined_output.txt
├── error.log
├── error_logger.py
├── gpu_selection.py
├── main_scriptv4JSONL.py  <-- Updated main script with re-mapping functionality
├── model_utils.py
├── pdf_ocr_utils.py
├── preprocessing.py
├── processed_files.log
├── processing.log
├── progress_tracker.py
├── __pycache__
│   ├── error_logger.cpython-312.pyc
│   ├── gpu_selection.cpython-312.pyc
│   ├── model_utils.cpython-312.pyc
│   ├── pdf_ocr_utils.cpython-312.pyc
│   ├── preprocessing.cpython-312.pyc
│   ├── progress_tracker.cpython-312.pyc
│   └── pymupdf_utils.cpython-312.pyc
├── pymupdf_utils.py
│
└── utils/
    ├── gui.py  <-- New GUI module for user interaction
    └── text_remapper.py  <-- New module for re-mapping functionality
```

**Flowchart:**
**Pseudocode:**

```
pdf_text_remapper/
│
├── combined_output.json
├── combined_output.py
├── combined_output.txt
├── error.log
├── error_logger.py
├── gpu_selection.py
├── main_scriptv4JSONL.py
│   ├── function main(input_dir, output_dir, output_format, mapping_config_path)
│   │   ├── Set TESSDATA_PREFIX and verify eng.traineddata file existence
│   │   ├── Load mapping configuration from JSON file
│   │   ├── Verify input and output directories exist
│   │   ├── Load processed files progress
│   │   ├── Filter PDF files to process
│   │   ├── Fetch GPU information, select a GPU, and set device
│   │   ├── Load model and processor for OCR text extraction
│   │   ├── Initialize skipped_files list
│   │   ├── For each pdf_file in pdf_files:
│   │   │   ├── Extract images and text using extract_images_and_text_ocr(pdf_path)
│   │   │   ├── Combine extracted text (pdf_text and ocr_text)
│   │   │   ├── If no text found, skip the file
│   │   │   ├── Apply re-mapping using remap_text(combined_text, mapping_config)
│   │   │   ├── Save output in selected format (txt, csv, or json) based on output_format
│   │   │   ├── Handle exceptions and log errors
│   │   │   ├── Update skipped_files list if processing fails
│   │   ├── Clean up GPU memory and display skipped files if any
│   │   └── Return None
│   └── If __name__ == "__main__":
│       │   ├── Parse command-line arguments using argparse
│       │   ├── Call main function with provided arguments
│       │   └── Return None
├── model_utils.py
├── pdf_ocr_utils.py
├── preprocessing.py
├── processed_files.log
├── processing.log
├── progress_tracker.py
├── __pycache__
│   ├── ... (Same as before)
└── pymupdf_utils.py
│
└── utils/
    ├── gui.py
│   ├── class PDFProcessingGUI:
│   │   ├── Initialize Tkinter GUI with input and output fields, and Start Processing button
│   │   ├── Function browse_input_dir():
│   │   │   ├── Open filedialog to select input directory
│   │   │   └── Update input_dir variable if a directory is selected
│   │   ├── Function browse_output_dir():
│   │   │   ├── Open filedialog to select output directory
│   │   │   └── Update output_dir variable if a directory is selected
│   │   ├── Function browse_mapping_config():
│   │   │   ├── Open filedialog to select mapping configuration file (JSON)
│   │   │   └── Update mapping_config_path variable if a file is selected
│   │   ├── Function start_processing():
│   │   │   ├── Validate user inputs and construct command for main_scriptv4JSONL.py
│   │   │   ├── Run the command using subprocess and handle exceptions
│   │   │   ├── Close GUI window after starting processing task
│   │   └── If __name__ == "__main__":
│   │       ├── Initialize PDFProcessingGUI instance
│   │       └── Start Tkinter event loop
    └── text_remapper.py
        ├── Function remap_text(text, mapping_config):
        │   ├── Initialize sections dictionary with empty lists as values
        │   ├── For each line in splitlines(text):
        │   │   ├── Set matched flag to False
        │   │   ├── For each section in mapping_config["sections"]:
        │   │   │   ├── If any keyword from the section matches the line, add it to the corresponding list and set matched flag to True
        │   │   │   └── Break the inner loop if a match is found
        │   │   └── If no match was found for the line, add it to the "Unmapped" list or create it if it doesn't exist
        │   ├── Return updated sections dictionary with remapped text content
        │   └── Return None
```

This pseudocode and flowchart provide a clear overview of the project structure and workflow, including the new re-mapping functionality and GUI-based user interaction. The pseudocode represents the main logic and functions of each module in a concise, tree-like format, making it easier to understand the overall process and how different components interact with one another.

To use this updated project structure and workflow, follow the provided code examples for each file and folder, ensuring that you update any necessary file paths or dependencies. After updating the project structure, run the GUI using the command:

```
python utils/gui.py
```

This will launch the PDF Text Remapper GUI, allowing users to interact with your processing pipeline through a graphical interface.
