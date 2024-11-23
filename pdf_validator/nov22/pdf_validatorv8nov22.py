import os

def validate_pdf(file_path):
    """
    Validates a PDF file by attempting to open it in binary mode 
    and checking for the PDF signature.

    Args:
      file_path: The path to the PDF file.

    Returns:
      True if the file appears to be a valid PDF, False otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            # Check for the PDF signature ("%PDF") at the beginning of the file
            if f.read(4) == b'%PDF':
                return True
            else:
                return False
    except:
        return False

def delete_files(file_paths):
    """
    Deletes a list of files.

    Args:
        file_paths: A list of file paths to delete.

    Returns:
        The number of files successfully deleted.
    """
    deleted_count = 0
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    return deleted_count

def scan_and_validate_pdf_files(directory_path, delete_invalid=False, recursive=False):
    """
    Scans a directory for PDF files, validates them, and optionally deletes invalid files.

    Args:
      directory_path: The path to the directory containing PDF files.
      delete_invalid: Whether to delete invalid PDF files.
      recursive: Whether to scan subdirectories recursively.

    Returns:
      A tuple containing:
        - A list of valid PDF file paths.
        - A list of invalid PDF file paths.
        - The number of invalid files deleted.
    """
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

        if not recursive:
            break

    deleted_count = 0
    if delete_invalid:
        deleted_count = delete_files(invalid_files)

    return valid_files, invalid_files, deleted_count

if __name__ == "__main__":
    target_directory = input("Enter the target directory path: ")

    recursive_option = input("Do you want to scan recursively? (Y/N): ")
    recursive_scan = recursive_option.strip().lower() == 'y'

    delete_option = input("Do you want to delete invalid/corrupted files? (Y/N): ")
    delete_invalid = delete_option.strip().lower() == 'y'

    print("\nScanning and validating PDF files...")
    valid_files, invalid_files, deleted_count = scan_and_validate_pdf_files(
        target_directory, delete_invalid, recursive_scan
    )

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

    print(f"\nSuggested file '{suggested_file_name}' with the list of "
          f"valid and invalid PDF files.")

    if delete_invalid:
        print(f"\nNumber of files deleted: {deleted_count}")
