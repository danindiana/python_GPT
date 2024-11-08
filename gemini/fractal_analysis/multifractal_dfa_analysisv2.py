import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt

def mfdfa(signal, q_vals):
    # Perform wavelet transform
    coeffs = pywt.wavedec(signal, 'db1')
    flucts = []
    
    for q in q_vals:
        if q == 0:
            fluct = np.exp(0.5 * np.mean(np.log(np.square(coeffs[-1]) + 1e-10)))  # Add small value to avoid log(0)
        else:
            fluct = np.mean(np.power(np.abs(coeffs[-1]) + 1e-10, q))**(1/q)  # Add small value to avoid division by zero
        flucts.append(fluct)
    
    return flucts

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

# Perform Multifractal DFA
q_vals = np.arange(-10, 11, 1)  # Range of q-values to analyze multifractal scaling
mfdfa_results = mfdfa(prices, q_vals)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(q_vals, mfdfa_results, marker='o', linestyle='-', color='b')
plt.title("Multifractal Detrended Fluctuation Analysis (MF-DFA)")
plt.xlabel("q-value")
plt.ylabel("Fluctuation Function F(q)")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()
