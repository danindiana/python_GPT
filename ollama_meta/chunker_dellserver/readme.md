System Specs: 
```
  OS: Debian 12 bookworm
    ,g$$P""       """Y$$.".      Kernel: x86_64 Linux 6.1.0-28-amd64
   ,$$P'              `$$$.      Uptime: 3h 42m
  ',$$P       ,ggs.     `$$b:    Packages: 2660
  `d$$'     ,$P"'   .    $$$     Shell: bash 5.2.15
   $$P      d$'     ,    $$P     Disk: 1.7T / 4.1T (43%)
   $$:      $$.   -    ,d$$'     CPU: Intel Xeon E5-1660 0 @ 12x 3.3GHz [70.0°C]
   $$\;      Y$b._   _,d$P'      GPU: NVIDIA GeForce GTX 1050 Ti, NVIDIA GeForce GTX 1060 6GB
   Y$$.    `.`"Y$$$$P"'          RAM: 3232MiB / 32047MiB
   `$$b      "-.__              
    `Y$$                        
     `Y$$.                      
       `$$b.                    
         `Y$$b.                 
            `"Y$b._             
                `""""           
                                
spook@dellserver:~
$ nvidia-smi 
Fri Dec 13 18:25:30 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1050 Ti     On  | 00000000:03:00.0 Off |                  N/A |
| 57%   85C    P0              N/A /  75W |   2528MiB /  4096MiB |     93%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce GTX 1060 6GB    On  | 00000000:04:00.0 Off |                  N/A |
| 33%   52C    P2              49W / 120W |   5226MiB /  6144MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2421      G   /usr/lib/xorg/Xorg                           30MiB |
|    0   N/A  N/A      2767      G   xfwm4                                         1MiB |
|    0   N/A  N/A    190709      C   ...unners/cuda_v12/ollama_llama_server     2492MiB |
|    1   N/A  N/A      2421      G   /usr/lib/xorg/Xorg                            4MiB |
|    1   N/A  N/A      7699      C   python                                     5218MiB |
+---------------------------------------------------------------------------------------+
tested and confirmed working on $ date
Fri Dec 13 06:25:51 PM CST 2024
```

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
