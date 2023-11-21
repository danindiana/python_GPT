This Python script is designed to extract text from PDF files, including using Optical Character Recognition (OCR) on images within those PDFs. It uses the pdfplumber and pytesseract libraries, with detailed logging of its operations. Here's an explanation of how it works:

Setting up pytesseract:

The script sets the command path for pytesseract to /usr/bin/tesseract, which is the default installation path for Tesseract OCR on most Unix systems.
Initializing Logging:

Logging is set up to display timestamps, logger name, log level, and the log message. This is useful for tracking the script's progress and debugging.
Prompting for Directories:

prompt_user_for_directories() asks the user to enter the paths of the source (where the PDF files are located) and target (where to save the extracted text files) directories.
Extracting Text from a PDF Page:

extract_text_from_page() extracts text from a given page using pdfplumber. If no text is found, it returns an empty string.
Extracting Text from Images:

extract_text_from_image() uses OCR (via pytesseract) to extract text from a PIL image object.
Processing the PDF Files:

extract_text_from_pdf() processes a PDF file by:
Opening the PDF with pdfplumber.
Iterating through each page, extracting text.
Converting each page to an image and extracting text from the image.
Combining text from all pages and saving it to a .txt file in the target directory.
Main Program Flow:

main() controls the overall program flow:
Retrieves source and target directory paths from the user.
Validates these directories and creates the target directory if it doesn't exist.
Lists all files in the source directory for logging.
Processes each PDF file found in the source directory.
Handles and logs any exceptions that occur, including a full traceback for debugging.
Error Handling and Logging:

Throughout the script, there's extensive error handling and logging to catch and report issues, like directory access problems, file reading errors, or issues in text extraction.
Running the Script:

When the script is executed, it calls main(). This starts the process of extracting text from all the PDF files in the specified source directory.
Overall, the script is a comprehensive tool for extracting text from PDFs, handling both regular text and text within images. It logs its operations thoroughly, making it easier to track its progress and troubleshoot any issues.
