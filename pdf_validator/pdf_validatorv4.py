import os
import fitz  # PyMuPDF's binding for working with PDFs

def validate_pdf(file_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        
        # Close the PDF file
        pdf_document.close()
        
        return True
    except:
        return False

def delete_files(file_paths):
    deleted_count = 0
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    return deleted_count

def scan_and_validate_pdf_files(directory_path, delete_invalid=False):
    valid_files = []
    invalid_files = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                print(f"Validating: {file_path}")
                if validate_pdf(file_path):
                    valid_files.append(file_path)
                    print(f"  Valid")
                else:
                    invalid_files.append(file_path)
                    print(f"  Invalid")
    
    deleted_count = 0
    if delete_invalid:
        deleted_count = delete_files(invalid_files)
    
    return valid_files, invalid_files, deleted_count

if __name__ == "__main__":
    target_directory = input("Enter the target directory path: ")
    
    delete_option = input("Do you want to delete invalid/corrupted files? (Y/N): ")
    delete_invalid = delete_option.strip().lower() == 'y'
    
    print("\nScanning and validating PDF files...")
    valid_files, invalid_files, deleted_count = scan_and_validate_pdf_files(target_directory, delete_invalid)
    
    print("\nValid PDF files:")
    for file_path in valid_files:
        print(file_path)
    
    print("\nInvalid PDF files:")
    for file_path in invalid_files:
        print(file_path)
    
    suggested_file_name = "validated_files.txt"
    with open(suggested_file_name, "w") as file:
        file.write("Valid PDF files:\n")
        file.writelines([f"{path}\n" for path in valid_files])
        file.write("\nInvalid PDF files:\n")
        file.writelines([f"{path}\n" for path in invalid_files])
    
    print(f"\nSuggested file '{suggested_file_name}' with the list of valid and invalid PDF files.")
    
    if delete_invalid:
        print(f"\nNumber of files deleted: {deleted_count}")
