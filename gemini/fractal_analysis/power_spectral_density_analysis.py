import os
import pandas as pd
import numpy as np
from scipy.signal import welch

# List CSV files in the current directory
csv_files = [f for f in os.listdir() if f.endswith('.csv')]

# Prompt user to select a file
print("Select a CSV file to analyze:")
for i, file in enumerate(csv_files):
    print(f"{i+1}: {file}")
file_index = int(input("Enter the number of the file: ")) - 1
file_path = csv_files[file_index]

# Load data
data = pd.read_csv(file_path)
prices = data['price'].values

# Power Spectral Density
frequencies, psd = welch(prices)
print("Frequencies:\n", frequencies)
print("Power Spectral Density:\n", psd)
