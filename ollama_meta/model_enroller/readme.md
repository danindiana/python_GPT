This Bash script automates the process of finding, organizing, and registering `.gguf` model files for use with **Ollama**, a tool for managing and running machine learning models. Here's a breakdown of what the script does:

---

### **1. Directory Setup**
- **Target Directory for Models**: The script defines a target directory (`/home/ollama/models`) where the `.gguf` model files will be stored.
- **Ollama Config Directory**: It also checks for the Ollama configuration directory (`~/.ollama`), which is where Ollama expects its configuration and model files.
- **Permissions**: The script ensures the target directory exists and sets the correct ownership (`$USER`) and permissions (`755`).

---

### **2. Symbolic Link Creation**
- If the `~/.ollama/models` directory does not already exist as a symbolic link, the script creates one pointing to the target directory (`/home/ollama/models`). This ensures Ollama can access the models stored in the target directory.

---

### **3. Search for `.gguf` Files**
- The script searches the entire filesystem (`/`) for files with the `.gguf` extension using the `find` command.
- If no `.gguf` files are found, the script exits with an error message.

---

### **4. Process Each `.gguf` File**
For each `.gguf` file found:
- **Model Directory**: A directory is created for the model under `/home/ollama/models`, named after the model (e.g., `/home/ollama/models/model_name`).
- **Copy File**: The `.gguf` file is copied to the model directory (if it doesn’t already exist there).
- **Create Modelfile**: A `Modelfile` is generated in the model directory. This file contains configuration parameters for the model, such as `temperature`, `top_p`, `repeat_penalty`, `top_k`, and `num_ctx`.
- **Register Model**: The model is registered with Ollama using the `ollama create` command and the generated `Modelfile`.

---

### **5. List Enrolled Models**
- After processing all `.gguf` files, the script lists the enrolled models using the `ollama list` command.

---

### **6. Completion Message**
- The script prints a message indicating that all `.gguf` files have been processed and enrolled.

---

### **Key Features**
- **Automation**: The script automates the entire process of finding, organizing, and registering `.gguf` models for Ollama.
- **Error Handling**: It skips files that already exist in the target directory and handles cases where no `.gguf` files are found.
- **Customization**: The `Modelfile` parameters can be adjusted to suit specific model requirements.

---

### **Example Workflow**
1. The script searches for `.gguf` files on the system.
2. For each file (e.g., `example_model.gguf`):
   - Creates a directory: `/home/ollama/models/example_model`.
   - Copies `example_model.gguf` to the directory.
   - Creates a `Modelfile` with predefined parameters.
   - Registers the model with Ollama.
3. Lists all registered models.

---

### **Potential Use Cases**
- Setting up a new system with multiple `.gguf` models for Ollama.
- Migrating or organizing existing `.gguf` files into a structured directory for Ollama.
- Automating the enrollment of models in a deployment pipeline.

---

### **Dependencies**
- **Ollama**: The script assumes Ollama is installed and available in the system's PATH.
- **Permissions**: The script uses `sudo` for directory creation and permission changes, so it requires elevated privileges.

---

This script is useful for users managing multiple machine learning models with Ollama, ensuring a consistent and automated setup process.

```
#!/bin/bash

# Define the target directory for Ollama models
OLLAMA_DIR="/home/ollama/models"
OLLAMA_CONFIG_DIR="$HOME/.ollama"

# Ensure the target directory exists with the correct permissions
sudo mkdir -p "$OLLAMA_DIR"
sudo chown -R "$USER":"$USER" "$OLLAMA_DIR"
chmod -R 755 "$OLLAMA_DIR"

# Optionally link the Ollama models directory to ~/.ollama
if [ ! -L "$OLLAMA_CONFIG_DIR/models" ]; then
    echo "Creating symbolic link for Ollama models directory..."
    rm -rf "$OLLAMA_CONFIG_DIR/models"
    ln -s "$OLLAMA_DIR" "$OLLAMA_CONFIG_DIR/models"
fi

# Find all .gguf files on the system
echo "Searching for .gguf files..."
GGUF_FILES=$(find / -type f -name "*.gguf" 2>/dev/null)

if [ -z "$GGUF_FILES" ]; then
    echo "No .gguf files found on the system."
    exit 1
fi

# Iterate over each .gguf file and enroll it
for GGUF_FILE in $GGUF_FILES; do
    echo "Processing file: $GGUF_FILE"

    # Extract the base name of the file (without directory and extension)
    MODEL_NAME=$(basename "$GGUF_FILE" .gguf)

    # Target directory for this model
    MODEL_DIR="$OLLAMA_DIR/$MODEL_NAME"

    # Create a directory for the model with correct permissions
    sudo mkdir -p "$MODEL_DIR"
    sudo chown -R "$USER":"$USER" "$MODEL_DIR"
    chmod -R 755 "$MODEL_DIR"

    # Copy the .gguf file to the model directory (prevent self-move errors)
    if [ ! -f "$MODEL_DIR/$(basename $GGUF_FILE)" ]; then
        cp "$GGUF_FILE" "$MODEL_DIR/"
    else
        echo "File $GGUF_FILE already exists in the target directory, skipping copy."
    fi

    # Create a Modelfile in the model directory
    MODELFILE="$MODEL_DIR/Modelfile"
    if [ ! -f "$MODELFILE" ]; then
        echo "Creating Modelfile for $MODEL_NAME..."
        cat > "$MODELFILE" <<EOF
# Modelfile for $MODEL_NAME
FROM ./${MODEL_NAME}.gguf

# Set parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.2
PARAMETER top_k 50
PARAMETER num_ctx 4096
EOF
    fi

    # Register the model with Ollama
    echo "Registering model $MODEL_NAME..."
    ollama create "$MODEL_NAME" -f "$MODELFILE" || echo "Failed to register $MODEL_NAME."
done

# List the available models
echo "Listing enrolled models:"
ollama list || echo "Failed to list models. Check permissions or logs."

echo "All .gguf files have been processed and enrolled."
```
The script does **not** use additional disk space when copying the `.gguf` files, provided that the source and destination directories are on the **same filesystem**. Here's why:

---

### **1. Copy Behavior**
- The script uses the `cp` command to copy `.gguf` files from their original location to the target directory (`/home/ollama/models`).
- If the source and destination are on the **same filesystem**, the `cp` command creates a **new copy** of the file, which **doubles the disk space usage** for that file.
- If the source and destination are on **different filesystems**, the file is physically copied, also using additional disk space.

---

### **2. Symbolic Link Alternative**
- The script creates a **symbolic link** (`ln -s`) for the `~/.ollama/models` directory to point to `/home/ollama/models`. This does **not** use additional disk space because symbolic links are just pointers to the original location.

---

### **3. Disk Space Optimization**
If you want to avoid using additional disk space, you can modify the script to **move** the files instead of copying them, or use **hard links**. Here's how:

#### **Option 1: Move Files Instead of Copying**
Replace the `cp` command with `mv`:
```bash
mv "$GGUF_FILE" "$MODEL_DIR/"
```
- This will **move** the file to the target directory, freeing up space in the original location.

#### **Option 2: Use Hard Links**
Replace the `cp` command with `ln`:
```bash
ln "$GGUF_FILE" "$MODEL_DIR/"
```
- This creates a **hard link**, which points to the same underlying data on disk. Both the original and the linked file will refer to the same data, so no additional disk space is used.
- Note: Hard links only work if the source and destination are on the **same filesystem**.

---

### **4. When Does the Script Use More Disk Space?**
The script uses more disk space **only if**:
- The `.gguf` files are copied (not moved or hard-linked).
- The source and destination are on the **same filesystem**.

---

### **Recommendation**
If disk space is a concern:
- Use **hard links** (if the files are on the same filesystem).
- Or **move** the files instead of copying them.

