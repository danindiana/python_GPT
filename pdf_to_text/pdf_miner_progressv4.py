import os
from pathlib import Path
from pdfminer.high_level import extract_text
from datetime import datetime
from tqdm import tqdm

def query_recursive_scan():
    """
    Ask the user if they want to recursively scan all files in the source directory.
    """
    choice = input("Do you want to recursively scan all files in the source directory? [y/n]: ").lower()
    return choice in ["y", "yes"]

def process_pdf_directory(current_dir, output_dir):
    """
    Process all PDF files in the given directory.
    """
    pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {current_dir}.")
        return

    print(f"Processing directory: {current_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_log = os.path.join(output_dir, f'processed_{os.path.basename(current_dir)}_{timestamp}.txt')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_file_log, 'w') as logfile:
        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            pdf_path = os.path.join(current_dir, filename)
            text_output_path = os.path.join(output_dir, filename) + '.txt'
            Path(os.path.dirname(text_output_path)).mkdir(parents=True, exist_ok=True)
            
            try:
                text = extract_text(pdf_path)
                with open(text_output_path, 'w') as text_file:
                    text_file.write(text)
                logfile.write(f"Processed {pdf_path}\n")
                print(f"Extracted text from {pdf_path} to {text_output_path}")
            except Exception as e:
                error_msg = f"Error extracting text from {pdf_path}: {e}"
                print(error_msg)
                logfile.write(f"{error_msg}\n")

def main():
    input_dir = 'your-input-directory'
    output_dir = 'your-output-directory'
    recursive = query_recursive_scan()

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"Source directory does not exist or is not a directory: {input_dir}")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created target directory: {output_dir}")
        except Exception as e:
            print(f"Unable to create target directory: {e}")
            return

    if recursive:
        for root, dirs, files in os.walk(input_dir):
            process_pdf_directory(root, output_dir)
    else:
        process_pdf_directory(input_dir, output_dir)

    print("Extraction completed. Check the target directory for the output files.")

if __name__ == "__main__":
    main()
