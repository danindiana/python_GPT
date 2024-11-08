import os
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

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
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
data.set_index('timestamp', inplace=True)

# Decompose time series
decomposition = seasonal_decompose(data['price'], model='additive', period=1)
print("Trend:\n", decomposition.trend.dropna())
print("\nSeasonal:\n", decomposition.seasonal.dropna())
print("\nResidual:\n", decomposition.resid.dropna())
