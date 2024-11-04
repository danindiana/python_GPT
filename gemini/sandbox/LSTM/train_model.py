from data_processing import load_data, preprocess_data, list_csv_files
from model_building import build_model

def train_model(model, X, y, batch_size=32, epochs=10):
    from tensorflow.keras.callbacks import TensorBoard
    
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

if __name__ == "__main__":
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        model = build_model(input_shape)
        
        train_model(model, X, y)

def main():
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        model = build_model(input_shape)
        
        train_model(model, X, y)

if __name__ == "__main__":
    main()
