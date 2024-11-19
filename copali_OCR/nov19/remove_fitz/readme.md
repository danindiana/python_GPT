This program is designed to process PDF files, extract text and images from them, and calculate similarity scores using a pre-trained model (ColQwen2). Here's a breakdown of what it does:

1. It checks for the necessary dependencies and Tesseract language data.
2. It defines functions for:
   - Preprocessing images for OCR
   - Extracting embedded text directly from PDFs
   - Extracting images and text using OCR if necessary
   - Splitting text into chunks
   - Fetching GPU information
   - Prompting the user to select a GPU
   - Loading and saving progress
   - Saving output in different formats (txt, json, csv)
3. It prompts the user for input and output directories and the desired output format.
4. It loads the progress log to keep track of processed files.
5. It filters the PDF files in the input directory that haven't been processed yet.
6. It prompts the user to resume from the last stop or start fresh.
7. It fetches GPU information and prompts the user to select a GPU.
8. It loads the ColQwen2 model and processor.
9. It sets a lower maximum chunk size for memory efficiency.
10. It processes each PDF file:
    - Extracts images and text using OCR if necessary
    - Saves the OCR-like text to a file in the output directory
    - Processes images in batches to prevent out-of-memory errors
    - Processes text chunks and calculates similarity scores
    - Skips files with long names, long chunks, or CUDA memory issues
    - Adds processed files to the progress log
11. It performs final memory cleanup.
12. It displays the list of skipped files, if any.

Overall, this program automates the process of extracting text and images from PDFs, calculating similarity scores using a pre-trained model, and saving the results in the desired format. It handles various edge cases and provides options for resuming progress and selecting a GPU.

Here's a refactored version of the program description in a bash-like tree structure:

```
.
├── Check dependencies and Tesseract language data
├── Define functions
│   ├── Preprocess images for OCR
│   ├── Extract embedded text from PDFs
│   ├── Extract images and text using OCR
│   ├── Split text into chunks
│   ├── Fetch GPU information
│   ├── Prompt user to select a GPU
│   ├── Load and save progress
│   └── Save output in different formats
├── Prompt user for input and output directories and output format
├── Load progress log
├── Filter unprocessed PDF files
├── Prompt user to resume or start fresh
├── Fetch GPU information and prompt user to select a GPU
├── Load ColQwen2 model and processor
├── Set maximum chunk size for memory efficiency
├── Process each PDF file
│   ├── Extract images and text using OCR
│   ├── Save OCR-like text to output file
│   ├── Process images in batches
│   ├── Process text chunks and calculate similarity scores
│   ├── Skip files with long names, long chunks, or CUDA memory issues
│   └── Add processed files to progress log
├── Perform final memory cleanup
└── Display list of skipped files, if any
```

This tree structure provides a high-level overview of the program's flow and main components.
