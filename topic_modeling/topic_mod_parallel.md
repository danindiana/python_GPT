Certainly! Let's break down the last output program step by step to understand how it works:

1. **Importing Libraries**:
   - The script starts by importing necessary libraries such as `os`, `re`, `pdfplumber`, `sklearn`, `pandas`, `tqdm`, and `joblib`.

2. **User Input for Directories**:
   - The script prompts the user to input the source directory (where the files are located) and the target directory (where the results will be saved).

3. **Creating the Target Directory**:
   - If the target directory does not already exist, the script creates it using `os.makedirs()`.

4. **Defining Helper Functions**:
   - **`sanitize_and_change_extension(filename)`**: This function cleans the filename by replacing invalid characters with underscores and changes the file extension to `.txt`.
   - **`read_text_file(file_path)`**: This function reads the content of a plain text file.
   - **`read_pdf_file(file_path)`**: This function extracts text from a PDF file using `pdfplumber`.

5. **Setting Batch Parameters**:
   - The script sets the `batch_size` to 500, which determines how many files will be processed at once.
   - It then lists all files in the source directory and calculates the total number of files.

6. **Processing Files in Batches**:
   - The script enters a `while` loop that continues until all files are processed.
   - For each batch, it selects a subset of files (`batch_files`) from the source directory.
   - It initializes empty lists `corpus` and `file_names` to store the text content and corresponding filenames.

7. **Reading and Processing Files**:
   - For each file in the batch, the script reads its content:
     - If the file is a PDF, it uses the `read_pdf_file` function.
     - If the file is a text file, it uses the `read_text_file` function.
   - If the file content is not empty, it appends the content to the `corpus` list and the filename to the `file_names` list.
   - If an error occurs during reading, it skips the file and prints an error message.

8. **Checking for Valid Files in Batch**:
   - If no valid files are found in the batch, it skips to the next batch.

9. **TF-IDF Vectorization**:
   - The script performs TF-IDF vectorization on the batch's corpus using `TfidfVectorizer`.

10. **NMF Topic Modeling**:
    - The script initializes an `NMF` model with the specified number of topics (`num_topics`), random state, and regularization parameters (`alpha_W` and `alpha_H`).
    - It fits the NMF model on the TF-IDF matrix (`tfidf_dtm`).

11. **Extracting Top Words**:
    - The script extracts the top words for each topic from the NMF model's components.

12. **Saving Results**:
    - For each file in the batch, the script determines the dominant topic and its top words.
    - It constructs a result text string and saves it to a file in the target directory.

13. **Updating Progress**:
    - The script updates the count of processed files and the batch number, then prints the progress.

14. **Completion Message**:
    - Once all batches are processed, the script prints a completion message.

This script efficiently processes large numbers of files in batches, performs topic modeling, and saves the results, making it suitable for handling large datasets.
