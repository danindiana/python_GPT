import pymupdf as fitz
import re
import os
import signal
import sys
import time
from urllib.parse import urlparse

# Global variables for suspend/resume and graceful shutdown
running = True
suspended = False
output_file = "extracted_references.txt"

def signal_handler(sig, frame):
    """Handles signals for suspend, resume, and shutdown."""
    global running, suspended
    if sig == signal.SIGUSR1:  # Suspend signal
        suspended = True
        print("\nSuspended. Press SIGUSR2 to resume.")
    elif sig == signal.SIGUSR2:  # Resume signal
        suspended = False
        print("\nResuming...")
    elif sig == signal.SIGINT or sig == signal.SIGTERM: #graceful shutdown
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

def process_pdf(pdf_file, output):
    """Processes a single PDF file."""
    extracted_references = extract_references_from_pdf(pdf_file)

    if extracted_references:
        output.write(f"\n\nReferences from: {pdf_file}\n")
        print(f"\nReferences from: {pdf_file}")

        for ref in extracted_references:
            output.write(f"{ref}\n")
            print(ref)

        urls = filter_urls(extracted_references)
        if urls:
            output.write("\nURLs:\n")
            print("\nURLs:")
            for url in urls:
                output.write(f"{url}\n")
                print(url)

        dois = filter_dois(extracted_references)
        if dois:
            output.write("\nDOIs:\n")
            print("\nDOIs:")
            for doi in dois:
                output.write(f"{doi}\n")
                print(doi)

    else:
        output.write(f"\nNo references found in: {pdf_file}\n")
        print(f"No references found in: {pdf_file}")

def process_directory(directory, output):
    """Recursively processes all PDF files in a directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_file = os.path.join(root, file)
                process_pdf(pdf_file, output)
                while suspended and running:
                    time.sleep(1) #pause.
                if not running:
                    return

# Set up signal handlers
signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGUSR2, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Get directory from user
directory = input("Enter the directory to scan: ")

try:
    with open(output_file, "a") as output:
        process_directory(directory, output)
except FileNotFoundError:
    print("Directory not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("Processing complete.")
