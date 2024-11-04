from data_processing import load_data, preprocess_data, list_csv_files
from model_building import build_model
import tensorflow as tf

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    return dataset

def train_model(model, dataset, epochs=10):
    from tensorflow.keras.callbacks import TensorBoard
    
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    
    model.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback])

def main():
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        
        # Adjust batch size for multi-GPU training
        batch_size = 32 * strategy.num_replicas_in_sync
        
        # Create dataset
        dataset = create_dataset(X, y, batch_size)
        
        # Build and compile the model within the strategy scope
        with strategy.scope():
            model = build_model(input_shape)
        
        train_model(model, dataset)

if __name__ == "__main__":
    main()
