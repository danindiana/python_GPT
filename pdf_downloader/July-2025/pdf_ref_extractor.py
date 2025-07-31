#!/usr/bin/env python3

import os
import re
import json
import argparse
import time
import sys
import subprocess
from urllib.parse import unquote

# This script can be run in two modes:
# 1. Manager Mode (default): Discovers PDFs and spawns worker processes for each.
# 2. Worker Mode (--process-file): Processes a single PDF file and prints JSON to stdout.

# --- Pre-compiled Regex Patterns ---
REFERENCE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\[\d+\]\s+.+",
        r"\(\w+ et al\., \d{4}\)",
        r"\w+ et al\., \d{4}",
        r"\w+, \d{4}",
        r"DOI:\s*10\.\S+",
        r"https?://\S+",
        r"arXiv:\d{4}\.\d+",
        r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*, \d{4}\b"
    ]
]
DOI_PATTERN = re.compile(r"10\.\S+")

# --- Core Processing Functions ---

def validate_pdf(file_path):
    """Validates if a file is a PDF by checking its header."""
    try:
        with open(file_path, 'rb') as f:
            return f.read(4) == b'%PDF'
    except IOError:
        return False

def extract_references_from_text(text):
    """Finds all potential references in a block of text."""
    references = []
    for pattern in REFERENCE_PATTERNS:
        references.extend(pattern.findall(text))
    
    unique_references = []
    seen = set()
    for ref in references:
        ref = ref.strip()
        if len(ref) > 10 and ref not in seen:
            seen.add(ref)
            unique_references.append(ref)
    return unique_references

def process_single_pdf(pdf_path):
    """
    Processes one PDF. This is the core logic for a worker.
    Returns a dictionary on success or a tuple on error.
    """
    # PyMuPDF is imported here so it's only needed in the worker process.
    import pymupdf as fitz

    if not validate_pdf(pdf_path):
        return {"error": "Invalid PDF header", "file": pdf_path}

    doc = None
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)

        if not text.strip():
            return None

        extracted_refs = extract_references_from_text(text)
        if not extracted_refs:
            return None

        urls = [ref for ref in extracted_refs if ref.startswith(('http:', 'https:'))]
        dois = [match.group(0) for ref in extracted_refs if (match := DOI_PATTERN.search(ref))]

        return {
            "file": pdf_path,
            "references": extracted_refs,
            "urls": list(set(urls)),
            "dois": list(set(dois))
        }
    finally:
        if doc:
            doc.close()

def run_as_worker(pdf_path):
    """
    Entry point for a worker process. It processes one file
    and prints the resulting JSON to standard output.
    """
    try:
        result = process_single_pdf(pdf_path)
        if result:
            json.dump(result, sys.stdout)
    except Exception as e:
        # If any unexpected error occurs, print it as a JSON error object.
        json.dump({"error": str(e), "file": pdf_path}, sys.stdout)

# --- Main Orchestration (Manager) ---

def run_as_manager(args):
    """
    Main function to discover PDFs and manage worker subprocesses.
    """
    if not os.path.isdir(args.directory):
        print(f"ERROR: Directory not found at '{args.directory}'")
        return

    print("INFO: Discovering PDF files...")
    pdf_files = [
        unquote(os.path.join(root, file))
        for root, _, files in os.walk(args.directory)
        for file in files
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("INFO: No PDF files found.")
        return

    print(f"INFO: Found {len(pdf_files)} PDF files. Starting processing.")
    output_filename = f"references_{time.strftime('%Y%m%d_%H%M%S')}.json"
    print(f"INFO: Results will be streamed to '{output_filename}'")

    error_files = []
    processed_count = 0
    
    with open(output_filename, "w", encoding='utf-8') as outfile:
        outfile.write("[\n")
        first_entry = True

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"--> Processing {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
            try:
                # Command to re-run this script as a worker for a single file.
                command = [sys.executable, __file__, "--process-file", pdf_path]
                
                # Launch the worker subprocess with a hard timeout.
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=60,  # Hard timeout of 60 seconds
                    check=False # Don't raise exception on non-zero exit code
                )

                if process.returncode != 0:
                    error_files.append(f"{pdf_path} (Worker exited with code {process.returncode}, stderr: {process.stderr.strip()})")
                    continue

                if not process.stdout:
                    continue # No output means no references found.

                result = json.loads(process.stdout)
                
                if "error" in result:
                    error_files.append(f"{pdf_path} ({result['error']})")
                else:
                    if not first_entry:
                        outfile.write(",\n")
                    json.dump(result, outfile, indent=4)
                    first_entry = False
                    processed_count += 1
                    outfile.flush()

            except subprocess.TimeoutExpired:
                error_files.append(f"{pdf_path} (Processing timed out after 60s)")
            except json.JSONDecodeError:
                error_files.append(f"{pdf_path} (Worker produced invalid JSON)")
            except KeyboardInterrupt:
                print("\nWARN: Keyboard interrupt received. Shutting down...")
                break
            except Exception as e:
                error_files.append(f"{pdf_path} (Manager error: {e})")

        outfile.write("\n]\n")

    print("\n--- Processing Summary ---")
    print(f"Successfully extracted references from {processed_count} files.")
    if error_files:
        print(f"\nEncountered {len(error_files)} errors or timeouts:")
        for f_error in error_files:
            print(f" - {f_error}")
    print(f"\nINFO: Results saved to '{output_filename}'. Script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts academic references from PDFs in a directory. "
                    "Uses a robust manager-worker pattern to handle hanging files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory", nargs='?', default=None,
        help="The directory to scan for PDF files (Manager mode)."
    )
    parser.add_argument(
        "--process-file",
        help="[INTERNAL] Process a single PDF file (Worker mode)."
    )
    
    args = parser.parse_args()

    if args.process_file:
        # If --process-file is used, run in Worker Mode.
        run_as_worker(args.process_file)
    elif args.directory:
        # Otherwise, run in Manager Mode.
        run_as_manager(args)
    else:
        parser.print_help()
