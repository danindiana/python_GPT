# process_filessmd.sh
#!/bin/bash

# Define directories
TARGET_DIR="your/home/files-2b-processed"
OUTPUT_DIR="output/dir/files"
CHUNK_SIZE=4000  # Number of words per chunk, adjust as needed
CONTEXT_WINDOW=8096  # Set your desired context window size

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over each .txt file in target directory
find "$TARGET_DIR" -type f -name "*.txt" | while read -r FILE_PATH; do
    echo "Processing $FILE_PATH"

    # Call the Python script with context window
    python3 chunk_process.py "$FILE_PATH" "$OUTPUT_DIR" "$CHUNK_SIZE" "$CONTEXT_WINDOW"
done
