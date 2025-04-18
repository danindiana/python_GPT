import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Autocorrelation
autocorrelation = [np.corrcoef(prices[:-lag], prices[lag:])[0, 1] for lag in range(1, 100)]
print("Autocorrelation Values:\n", autocorrelation)
