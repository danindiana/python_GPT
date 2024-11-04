import os
from data_processing import load_data, preprocess_data, list_csv_files
from model_building import build_model
from training_model import train_model

def save_trained_model(model, file_name='lstm_model.keras'):
    from tensorflow.keras.models import save_model
    
    save_model(model, file_name)

if __name__ == "__main__":
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        model = build_model(input_shape)
        
        # Train the model
        train_model(model, X, y)
        
        # Save the trained model in native Keras format
        save_trained_model(model)

def main():
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        model = build_model(input_shape)
        
        # Train the model
        train_model(model, X, y)
        
        # Save the trained model in native Keras format
        save_trained_model(model)

if __name__ == "__main__":
    main()
