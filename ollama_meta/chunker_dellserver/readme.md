This script is a tool designed to process text files by sending chunks of text to an API (Ollama) for processing, saving the results to an output directory. Here's how it works, step by step:

---

### **1. Configuration Section**
The script sets up global constants:
- `OLLAMA_URL`: URL for the Ollama API endpoint.
- `MODEL_NAME`: Name of the model used for processing text.
- `CHUNK_SIZE`: Maximum size of each chunk of text sent to the API.
- `DEFAULT_OUTPUT_BASE`: Default directory for saving processed output.

---

### **2. `process_text_with_ollama`**
This function interacts with the API:
- **Input**: A text chunk (`prompt`).
- **Payload**: Constructs a JSON payload with the prompt, model name, and additional parameters like `temperature` and `max_tokens`.
- **API Call**: Sends a POST request to the API and handles potential errors:
  - Logs the raw response for debugging.
  - Parses the response as JSON and extracts the "response" field.
- **Error Handling**:
  - Logs and handles connection errors.
  - Handles malformed JSON errors (e.g., non-JSON responses).

---

### **3. `chunk_text`**
Splits a large text file into smaller chunks for processing:
- Uses Python generators to efficiently yield chunks of text without loading everything into memory.

---

### **4. `process_files`**
Processes all `.txt` files in the specified input directory:
- Iterates over files in `input_dir` and skips non-`.txt` files.
- Reads the file content and splits it into chunks using `chunk_text`.
- Sends each chunk to `process_text_with_ollama`.
- Logs and stores processed chunks in a list.
- Saves the combined output of processed chunks into a file in `output_dir`.
- Logs the saved file content to the console.

---

### **5. `check_permissions`**
Checks if the script has the required permissions (`read` or `write`) for a given directory.

---

### **6. `ensure_output_directory`**
Ensures the output directory exists and is writable:
- Creates the directory if it doesn’t exist.
- Verifies write permissions.
- Logs success or errors.

---

### **7. `main`**
The main function orchestrates the workflow:
1. **Prompt for Input Directory**: 
   - Asks the user to specify the directory containing text files.
   - Validates the directory's existence and read permissions.
2. **Prompt for Output Directory**:
   - Lets the user specify an output directory or defaults to a timestamped subdirectory in `DEFAULT_OUTPUT_BASE`.
   - Ensures the output directory is ready for use.
3. **Process Files**:
   - Calls `process_files` to process all `.txt` files in the input directory and save their processed outputs.
4. **Completion Message**:
   - Prints a message indicating where the processed files were saved.

---

### **Key Features**
- **Chunk Processing**: Handles large text files by breaking them into manageable chunks.
- **Error Handling**: Includes robust handling for connection issues, JSON decoding errors, and permission issues.
- **Logging**: Logs raw API responses, processed chunks, and final outputs for debugging.
- **User Interaction**: Prompts the user to specify input/output directories.

---

### **Usage**
1. Run the script: 
   ```bash
   python script_name.py
   ```
2. Enter the input directory containing text files when prompted.
3. Specify an output directory or press Enter to use the default.
4. The script processes the files and saves the outputs, displaying logs for each step.

---

### **Debugging Enhancements**
- Logs raw API responses to the console for troubleshooting.
- Handles and displays errors related to API connectivity or malformed JSON responses.

Let me know if you'd like further clarification or adjustments!
