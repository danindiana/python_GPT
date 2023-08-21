import os
import numpy as np
from scipy.special import rel_entr

def find_text_files(directory):
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.txt'):
                text_files.append(os.path.join(root, file))
    return text_files

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

def main():
    root_directory = input("Enter the root directory to start scanning: ")
    target_text_length = int(input("Enter the target text length for preprocessing: "))

    text_files = find_text_files(root_directory)

    if not text_files:
        print("No text files found.")
        return

    print("Text files found:")
    for idx, file in enumerate(text_files, start=1):
        print(f"{idx}.) {file}")

    selected_input = input("Select files by entering numbers/ranges (e.g., 1-20,34,36,55,62-76,87,93-120): ")
    selected_indices = parse_selection(selected_input)

    selected_texts = []
    selected_filenames = []
    for idx in selected_indices:
        if 0 <= idx - 1 < len(text_files):
            with open(text_files[idx - 1], "r") as file:
                text = file.read()
                preprocessed_text = preprocess_text(text, target_text_length)
                selected_texts.append(preprocessed_text)
                selected_filenames.append(text_files[idx - 1])

    if len(selected_texts) < 2:
        print("At least 2 text files are required for JS divergence calculation.")
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
