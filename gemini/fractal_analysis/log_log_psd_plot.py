import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import matplotlib

# Use an interactive backend
matplotlib.use('Qt5Agg')

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

# Calculate Power Spectral Density (PSD)
frequencies, psd = welch(prices)

# Plot PSD on a log-log scale
plt.figure(figsize=(10, 6))
plt.loglog(frequencies[1:], psd[1:], marker='o', linestyle='-', color='b')
plt.title("Power Spectral Density (PSD) on Log-Log Scale")
plt.xlabel("Frequency (log scale)")
plt.ylabel("Power Spectral Density (log scale)")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()
