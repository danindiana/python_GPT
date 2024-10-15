import os
import logging
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF's binding for working with PDFs
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ValidationResult(Enum):
    VALID = "Valid"
    INVALID = "Invalid"

def validate_pdf(file_path: Path) -> bool:
    try:
        # Open the PDF file
        with fitz.open(file_path) as pdf_document:
            pass  # No need to do anything else if opening is successful
        return True
    except fitz.FileDataError:
        logging.error(f"File data error: {file_path}")
        return False
    except Exception as e:
        logging.error(f"Error validating {file_path}: {e}")
        return False

def delete_files(file_paths: List[Path]) -> int:
    deleted_count = 0
    for file_path in file_paths:
        try:
            file_path.unlink()
            logging.info(f"Deleted: {file_path}")
            deleted_count += 1
        except FileNotFoundError:
            logging.warning(f"File not found: {file_path}")
        except PermissionError:
            logging.error(f"Permission denied: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")
    return deleted_count

def scan_and_validate_pdf_files(directory_path: Path, delete_invalid: bool = False) -> Tuple[List[Path], List[Path], int]:
    valid_files = []
    invalid_files = []
    
    for file_path in directory_path.rglob("*.pdf"):
        logging.info(f"Validating: {file_path}")
        if validate_pdf(file_path):
            valid_files.append(file_path)
            logging.info(f"  {ValidationResult.VALID.value}")
        else:
            invalid_files.append(file_path)
            logging.info(f"  {ValidationResult.INVALID.value}")
    
    deleted_count = 0
    if delete_invalid:
        deleted_count = delete_files(invalid_files)
    
    return valid_files, invalid_files, deleted_count

def main():
    target_directory = Path(input("Enter the target directory path: "))
    
    delete_option = input("Do you want to delete invalid/corrupted files? (Y/N): ")
    delete_invalid = delete_option.strip().lower() == 'y'
    
    logging.info("Scanning and validating PDF files...")
    valid_files, invalid_files, deleted_count = scan_and_validate_pdf_files(target_directory, delete_invalid)
    
    logging.info("\nValid PDF files:")
    for file_path in valid_files:
        logging.info(file_path)
    
    logging.info("\nInvalid PDF files:")
    for file_path in invalid_files:
        logging.info(file_path)
    
    suggested_file_name = "validated_files.txt"
    with open(suggested_file_name, "w") as file:
        file.write("Valid PDF files:\n")
        file.writelines([f"{path}\n" for path in valid_files])
        file.write("\nInvalid PDF files:\n")
        file.writelines([f"{path}\n" for path in invalid_files])
    
    logging.info(f"\nSuggested file '{suggested_file_name}' with the list of valid and invalid PDF files.")
    
    if delete_invalid:
        logging.info(f"\nNumber of files deleted: {deleted_count}")

if __name__ == "__main__":
    main()
