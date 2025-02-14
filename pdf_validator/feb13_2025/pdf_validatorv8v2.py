import os
import urllib.parse
import sys

def validate_pdf(file_path):
    """Validates a PDF file."""
    try:
        with open(file_path, 'rb') as f:
            if f.read(4) == b'%PDF':
                return True
            else:
                return False
    except Exception:
        return False

def delete_files(file_paths):
    """Deletes a list of files."""
    deleted_count = 0
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)}: {e}")
    return deleted_count

def scan_and_validate_pdf_files(directory_path, delete_invalid=False, recursive=False):
    """Scans, validates, and optionally deletes PDF files."""
    valid_files = []
    invalid_files = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)

                try:
                    decoded_file_path = urllib.parse.unquote(file_path)
                except Exception as e:
                    print(f"Error decoding filename {file_path}: {e}")
                    invalid_files.append(file_path)
                    continue

                print(f"Validating: {decoded_file_path.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)}")

                if validate_pdf(decoded_file_path):
                    valid_files.append(decoded_file_path)
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
        print(file_path.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))  # Encode for printing

    print("\nInvalid PDF files:")
    for file_path in invalid_files:
        print(file_path.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))  # Encode for printing


    suggested_file_name = "validated_files.txt"
    with open(suggested_file_name, "w", encoding="utf-8") as file: # Explicit UTF-8 encoding for file writing
        file.write("Valid PDF files:\n")
        file.writelines([f"{path}\n" for path in valid_files])
        file.write("\nInvalid PDF files:\n")
        file.writelines([f"{path}\n" for path in invalid_files])

    print(f"\nSuggested file '{suggested_file_name}' with the list of "
          f"valid and invalid PDF files.")

    if delete_invalid:
        print(f"\nNumber of files deleted: {deleted_count}")
