import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.sort_values(by='timestampms')

def preprocess_data(df, seq_length=5):
    features = df[['price', 'amount']].values
    target = df['price'].values
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(features, seq_length)
    return X, y, scaler

def list_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the directory.")
        return None
    
    print("Available CSV files:")
    for i, file_name in enumerate(csv_files):
        print(f"{i+1}. {file_name}")
    
    choice = input("Enter the number of the file you want to load: ")
    try:
        choice = int(choice)
        if 1 <= choice <= len(csv_files):
            return os.path.join(directory, csv_files[choice-1])
        else:
            print("Invalid choice. Please enter a valid number.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None

if __name__ == "__main__":
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        print("Data preprocessing complete.")
