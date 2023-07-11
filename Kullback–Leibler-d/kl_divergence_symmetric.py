import numpy as np
from collections import Counter
import os
import PyPDF2

def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log(p / q))

def normalize(counter):
    total = sum(counter.values())
    return {char: count/total for char, count in counter.items()}

def preprocess_text(text):
    text = text.lower()
    counter = Counter(text)
    return normalize(counter)

def preprocess_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return preprocess_text(text)

def get_files(directory):
    files = os.listdir(directory)
    return [file for file in files if file.endswith('.txt') or file.endswith('.pdf')]

def select_files(directory, files):
    print("Please select files to compare:")
    for i, file in enumerate(files):
        print(f"{i+1}: {file}")
    indices = input("Enter the numbers of the files (separated by commas): ")
    indices = [int(index.strip()) - 1 for index in indices.split(',')]
    selected_files = [files[index] for index in indices]
    return [os.path.join(directory, file) for file in selected_files]

def main():
    directory = input("Enter the directory to scan for files: ")

    files = get_files(directory)
    print(f"Found {len(files)} files in {directory}.")

    selected_files = select_files(directory, files)

    distributions = []
    for file in selected_files:
        if file.endswith('.txt'):
            with open(file, 'r', encoding='utf-8-sig') as f:
                text = f.read()
            dist = preprocess_text(text)
        elif file.endswith('.pdf'):
            dist = preprocess_pdf(file)
        distributions.append(dist)

    num_files = len(selected_files)
    divergence_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(i, num_files):
            dist1 = distributions[i]
            dist2 = distributions[j]

            # Fill zero values for unseen characters in each distribution
            for key in (dist1.keys() - dist2.keys()):
                dist2[key] = 0
            for key in (dist2.keys() - dist1.keys()):
                dist1[key] = 0

            # Convert the distributions into numpy arrays
            p = np.array(list(dist1.values()))
            q = np.array(list(dist2.values()))

            divergence_pq = kl_divergence(p, q)
            divergence_qp = kl_divergence(q, p)

            divergence_matrix[i, j] = divergence_pq
            divergence_matrix[j, i] = divergence_qp

    for i in range(num_files):
        for j in range(i+1, num_files):
            file1 = selected_files[i]
            file2 = selected_files[j]
            divergence_pq = divergence_matrix[i, j]
            divergence_qp = divergence_matrix[j, i]
            print(f"Kullback-Leibler divergence from {file1} to {file2}: {divergence_pq}")
            print(f"Kullback-Leibler divergence from {file2} to {file1}: {divergence_qp}")

if __name__ == "__main__":
    main()
