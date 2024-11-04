import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(X, y, batch_size, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def create_eval_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def train_model(model, dataset, epochs=10, steps_per_epoch=None):
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])

def evaluate_model(model, dataset):
    predictions = model.predict(dataset)
    return predictions

def save_trained_model(model, file_name='lstm_model.keras'):
    save_model(model, file_name)

def main():
    check_gpu_availability()
    
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
        
        # Calculate steps per epoch
        steps_per_epoch = len(X) // batch_size
        
        # Build and compile the model within the strategy scope
        with strategy.scope():
            model = build_model(input_shape)
        
        # Train the model
        train_model(model, dataset, steps_per_epoch=steps_per_epoch)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create evaluation dataset
        eval_dataset = create_eval_dataset(X_test, y_test, batch_size)
        
        # Evaluate the model
        predictions = evaluate_model(model, eval_dataset)
        
        # Ensure that both y_test and predictions have the same shape
        if y_test.ndim == 2:
            y_test = y_test[:, 0]
        
        if predictions.ndim == 2:
            predictions = predictions[:, 0]
        
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        
        # Save the trained model in native Keras format
        save_trained_model(model)

if __name__ == "__main__":
    main()
