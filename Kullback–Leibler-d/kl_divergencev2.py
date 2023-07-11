import numpy as np
from collections import Counter
import os

def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * np.log(p / q))


def normalize(counter):
    total = sum(counter.values())
    return {char: count/total for char, count in counter.items()}

def preprocess(text):
    text = text.lower()
    counter = Counter(text)
    return normalize(counter)

def get_text_files(directory):
    files = os.listdir(directory)
    return [file for file in files if file.endswith('.txt')]

def select_file(directory, text_files):
    print("Please select a file:")
    for i, file in enumerate(text_files):
        print(f"{i+1}: {file}")
    index = int(input("Enter the number of the file: ")) - 1
    return os.path.join(directory, text_files[index])

def main():
    directory = input("Enter the directory to scan for text files: ")

    text_files = get_text_files(directory)
    print(f"Found {len(text_files)} text files in {directory}.")

    file1 = select_file(directory, text_files)
    file2 = select_file(directory, text_files)

    with open(file1, 'r', encoding='utf-8-sig') as f:
        text1 = f.read()
    with open(file2, 'r', encoding='utf-8-sig') as f:
        text2 = f.read()

    dist1 = preprocess(text1)
    dist2 = preprocess(text2)

    # Fill zero values for unseen characters in each distribution
    for key in (dist1.keys() - dist2.keys()):
        dist2[key] = 0
    for key in (dist2.keys() - dist1.keys()):
        dist1[key] = 0

    # Convert the distributions into numpy arrays
    p = np.array(list(dist1.values()))
    q = np.array(list(dist2.values()))

    print(f"Kullback-Leibler divergence between {file1} and {file2}:", kl_divergence(p, q))

if __name__ == "__main__":
    main()
