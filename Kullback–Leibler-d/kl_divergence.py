import numpy as np
from collections import Counter
import os

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def normalize(counter):
    total = sum(counter.values())
    return {char: count/total for char, count in counter.items()}

def preprocess(text):
    text = text.lower()
    counter = Counter(text)
    return normalize(counter)

def main():
    file1 = input("Enter the path of the first file: ")
    file2 = input("Enter the path of the second file: ")

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

    print("Kullback-Leibler divergence:", kl_divergence(p, q))

if __name__ == "__main__":
    main()
