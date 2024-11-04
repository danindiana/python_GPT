from data_processing import load_data, preprocess_data, list_csv_files
from model_building import build_model
import tensorflow as tf

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

def train_model(model, X, y, batch_size=32, epochs=10):
    from tensorflow.keras.callbacks import TensorBoard
    
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    
    # Adjust batch size for multi-GPU training
    batch_size = batch_size * strategy.num_replicas_in_sync
    
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

def main():
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        
        # Build and compile the model within the strategy scope
        with strategy.scope():
            model = build_model(input_shape)
        
        train_model(model, X, y)

if __name__ == "__main__":
    main()
