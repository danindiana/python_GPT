import os
import numpy as np
from scipy.special import rel_entr
import fitz  # PyMuPDF library for PDF processing

def find_pdf_files(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def preprocess_text(text, target_length):
    text = text[:target_length].ljust(target_length)
    return text

def calculate_js_divergence(p, q):
    m = 0.5 * (p + q)
    js_divergence = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
    return js_divergence

def parse_selection(selection):
    ranges = selection.split(',')
    selected_indices = []

    for rng in ranges:
        if '-' in rng:
            start, end = map(int, rng.split('-'))
            selected_indices.extend(range(start, end + 1))
        else:
            selected_indices.append(int(rng))
    
    return selected_indices

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def convert_windows_path_to_wsl_path(windows_path):
    wsl_path = windows_path.replace("C:\\", "/mnt/c/").replace("\\", "/")
    return wsl_path

def main():
    root_directory = input("Enter the root directory to start scanning (Windows-style path): ")
    target_text_length = int(input("Enter the target text length for preprocessing: "))

    root_directory_wsl = convert_windows_path_to_wsl_path(root_directory)
    pdf_files = find_pdf_files(root_directory_wsl)

    if not pdf_files:
        print("No PDF files found.")
        return

    print("PDF files found:")
    for idx, file in enumerate(pdf_files, start=1):
        print(f"{idx}.) {file}")

    selected_input = input("Select files by entering numbers/ranges (e.g., 1-20,34,36,55,62-76,87,93-120): ")
    selected_indices = parse_selection(selected_input)

    selected_texts = []
    selected_filenames = []
    for idx in selected_indices:
        if 0 <= idx - 1 < len(pdf_files):
            pdf_path_wsl = pdf_files[idx - 1]
            text = extract_text_from_pdf(pdf_path_wsl)
            preprocessed_text = preprocess_text(text, target_text_length)
            selected_texts.append(preprocessed_text)
            selected_filenames.append(pdf_path_wsl)

    if len(selected_texts) < 2:
        print("At least 2 PDF files are required for JS divergence calculation.")
        return

    print("\nSelected files:")
    for idx, (text, filename) in enumerate(zip(selected_texts, selected_filenames), start=1):
        print(f"{idx}.) {filename}\n{text}")

    p = np.array([ord(c) for c in selected_texts[0]])
    q_list = [np.array([ord(c) for c in text]) for text in selected_texts[1:]]

    js_divergences = []
    for q in q_list:
        js_divergence = calculate_js_divergence(p, q)
        js_divergences.append(js_divergence)

    print("\nJensen-Shannon Divergence:")
    for idx, (divergence, filename) in enumerate(zip(js_divergences, selected_filenames[1:]), start=1):
        print(f"{idx}.) {filename}: {divergence:.4f}")

    sort_choice = input("\nDo you want to sort JS divergence scores from highest to lowest? (y/n): ")
    if sort_choice.lower() == "y":
        sorted_indices = sorted(range(len(js_divergences)), key=lambda k: js_divergences[k], reverse=True)
        print("\nSorted Jensen-Shannon Divergence:")
        for idx, sorted_idx in enumerate(sorted_indices, start=1):
            print(f"{idx}.) {selected_filenames[sorted_idx + 1]}: {js_divergences[sorted_idx]:.4f}")

    choice = input("\nDo you want to go again? (yes/no): ")
    if choice.lower() == "yes":
        main()
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
