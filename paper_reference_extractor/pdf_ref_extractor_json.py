import pymupdf as fitz
import re
import os
import signal
import sys
import time
from urllib.parse import urlparse
import json

# Global variables for suspend/resume and graceful shutdown
running = True
suspended = False
output_file = "extracted_references"  # Base name, extension will be added
output_format = "txt"  # Default format

def signal_handler(sig, frame):
    """Handles signals for suspend, resume, and shutdown."""
    global running, suspended
    if sig == signal.SIGUSR1:  # Suspend signal
        suspended = True
        print("\nSuspended. Press SIGUSR2 to resume.")
    elif sig == signal.SIGUSR2:  # Resume signal
        suspended = False
        print("\nResuming...")
    elif sig == signal.SIGINT or sig == signal.SIGTERM:  # graceful shutdown
        running = False
        print("\nGraceful shutdown initiated. Finishing current file and exiting.")

def extract_references_from_pdf(pdf_path):
    """Extracts potential references from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        reference_patterns = [
            r"\[\d+\]\s+.+",
            r"\(\w+ et al., \d{4}\)",
            r"\w+ et al., \d{4}",
            r"\w+, \d{4}",
            r"DOI:\s*10.\S+",
            r"https?://\S+",
            r"arXiv:\d{4}\.\d+",
            r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*, \d{4}\b"
        ]

        references = []
        for pattern in reference_patterns:
            references.extend(re.findall(pattern, text))

        unique_references = []
        seen = set()
        for ref in references:
            ref = ref.strip()
            if ref not in seen:
                seen.add(ref)
                if len(ref) > 10:
                    unique_references.append(ref)

        return unique_references

    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def filter_urls(references):
    """Filters a list of references, returning only URLs."""
    urls = []
    for ref in references:
        try:
            result = urlparse(ref)
            if all([result.scheme, result.netloc]):
                urls.append(ref)
        except:
            pass
    return urls

def filter_dois(references):
    """Filters a list of references, returning only DOIs."""
    dois = []
    doi_pattern = r"10.\S+"
    for ref in references:
        match = re.search(doi_pattern, ref)
        if match:
            dois.append(match.group(0))
    return dois

def process_pdf(pdf_file, output_data):
    """Processes a single PDF file and adds data to the output dictionary."""
    extracted_references = extract_references_from_pdf(pdf_file)

    if extracted_references:
        output_data[pdf_file] = {
            "references": extracted_references,
            "urls": filter_urls(extracted_references),
            "dois": filter_dois(extracted_references)
        }
        print(f"\nReferences from: {pdf_file}")
        for ref in extracted_references:
            print(ref)
        print("\nURLs:")
        for url in output_data[pdf_file]["urls"]:
            print(url)
        print("\nDOIs:")
        for doi in output_data[pdf_file]["dois"]:
            print(doi)

    else:
        output_data[pdf_file] = "No references found."
        print(f"No references found in: {pdf_file}")

def process_directory(directory, output_data):
    """Recursively processes all PDF files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_file = os.path.join(root, file)
                process_pdf(pdf_file, output_data)
                while suspended and running:
                    time.sleep(1)  # pause.
                if not running:
                    return

# Set up signal handlers
signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGUSR2, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Get directory from user
directory = input("Enter the directory to scan: ")

# Get output format from user
output_format = input("Enter output format (txt or json): ").lower()
while output_format not in ("txt", "json"):
    output_format = input("Invalid format. Enter txt or json: ").lower()

try:
    output_data = {}
    process_directory(directory, output_data)

    if running: #only write if not interrupted.
        if output_format == "txt":
            with open(output_file + ".txt", "w") as output:
                for pdf_path, data in output_data.items():
                    output.write(f"\n\nReferences from: {pdf_path}\n")
                    if isinstance(data, dict):
                        for ref in data.get("references", []):
                            output.write(f"{ref}\n")
                        output.write("\nURLs:\n")
                        for url in data.get("urls", []):
                            output.write(f"{url}\n")
                        output.write("\nDOIs:\n")
                        for doi in data.get("dois", []):
                            output.write(f"{doi}\n")
                    else:
                        output.write(f"{data}\n")
        elif output_format == "json":
            with open(output_file + ".json", "w") as output:
                json.dump(output_data, output, indent=4)

except FileNotFoundError:
    print("Directory not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

if running: #only print if not interrupted.
    print("Processing complete.")
