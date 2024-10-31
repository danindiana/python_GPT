#!/bin/bash

# Define directories
TARGET_DIR="/ur/target/dir/"
OUTPUT_DIR="/ur/destination/dir"
CHUNK_SIZE=5000  # Number of words per chunk, adjust as needed

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over each .txt file in target directory
find "$TARGET_DIR" -type f -name "*.txt" | while read -r FILE_PATH; do
    echo "Processing $FILE_PATH"
    
    # Call the Python script for chunking and processing
    python3 chunk_process.py "$FILE_PATH" "$OUTPUT_DIR" "$CHUNK_SIZE"
done
