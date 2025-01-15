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
