This program is a batch processing script for analyzing and extracting topics from a collection of text documents, which can be in either plain text or PDF format. It follows these steps:

Source and Target Directories: It specifies the source directory where the input documents are located (source_directory) and the target directory where the output results will be saved (target_directory).

Directory Creation: It checks if the target directory exists and creates it if it doesn't.

File Sanitization: It defines a function sanitize_and_change_extension to sanitize filenames, replacing characters that are not allowed in filenames with underscores and changing the file extension to .txt.

Text Extraction Functions: It defines two functions for reading text from files:

read_text_file: Reads text from a plain text file.
read_pdf_file: Extracts text from a PDF file using the pdfplumber library.
Batch Processing: The script processes documents in batches to avoid memory issues:

It defines a batch_size to specify how many files to process in each batch.
It lists the files in the source directory.
It initializes counters for total processed files, current batch number, and processed files within the batch.
Main Loop: The main loop runs as long as there are unprocessed files in the source directory:

It selects a batch of files based on the batch_size.
For each file in the batch, it reads the content using the appropriate function (read_text_file for text files, read_pdf_file for PDF files) and adds it to a corpus if text is successfully extracted.
It keeps track of the file names for further processing.
TF-IDF Vectorization: After processing the batch, it performs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization on the corpus of text data. This step converts the text into a numerical representation suitable for topic modeling.

Topic Modeling (NMF): It applies Non-Negative Matrix Factorization (NMF) for topic modeling on the TF-IDF matrix. This step identifies topics in the corpus and assigns each document to one or more topics.

Top Words for Topics: It extracts the top words associated with each topic, which represent the most significant terms in those topics.

Saving Results: For each document in the batch, it saves the following information in a result file in the target directory:

The sanitized filename.
The assigned topic.
The top words for the document's topic.
Batch Completion: It updates counters and moves on to the next batch until all files are processed.

Completion Message: Once all batches are processed, it prints a message indicating that all batches have been processed.

Dependencies:

os: Standard Python library for interacting with the operating system.
re: Regular expression library for text manipulation.
pdfplumber: A library for extracting text from PDF files.
sklearn: Scikit-learn, a machine learning library for TF-IDF vectorization and NMF topic modeling.
pandas: A data manipulation library used for organizing and saving the results.
Please ensure that these libraries are installed in your Python environment for the program to work correctly.
