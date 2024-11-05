import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Flatten
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

def check_gpu_availability():
    logging.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.sort_values(by='timestampms')
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return None
    except pd.errors.ParserError:
        logging.error("Invalid CSV file: %s", file_path)
        return None

def explore_data(df):
    # Summary statistics
    logging.info("\n%s", df.describe())

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestampms'], df['price'])
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Price Over Time')
    plt.show()

def preprocess_data(df, seq_length=5, window_size=3):
    # Add technical indicators
    df['price_ma'] = df['price'].rolling(window=window_size).mean()
    df['price_volatility'] = df['price'].rolling(window=window_size).std()
    df['price_momentum'] = df['price'].diff(window_size)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    df = df.dropna()
    
    features = df[['price', 'amount', 'price_ma', 'price_volatility', 'price_momentum', 'rsi']].values
    target = df['price'].values
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length - 1][0])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(features, seq_length)
    logging.info("Generated X shape: %s", X.shape)
    logging.info("Generated y shape: %s", y.shape)
    return X, y, scaler

def list_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        logging.warning("No CSV files found in the directory.")
        return None
    
    logging.info("Available CSV files:")
    for i, file_name in enumerate(csv_files):
        logging.info("%d. %s", i+1, file_name)
    
    while True:
        try:
            choice = int(input("Enter the number of the file you want to load: "))
            if 1 <= choice <= len(csv_files):
                return os.path.join(directory, csv_files[choice-1])
            else:
                logging.warning("Invalid choice. Please enter a valid number.")
        except ValueError:
            logging.warning("Invalid input. Please enter a number.")

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(250, return_sequences=True, dropout=0.2)(inputs)
    lstm2 = LSTM(250, return_sequences=True, dropout=0.2)(lstm1)
    attention = Attention()([lstm2, lstm2])
    flatten = Flatten()(attention)
    dense1 = Dense(200, activation='relu')(flatten)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(100, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    outputs = Dense(1)(dropout2)
    
    model = Model(inputs=inputs, outputs=outputs)
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

def train_model(model, train_dataset, val_dataset, epochs=60, steps_per_epoch=None):
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, early_stopping],
        verbose=2
    )

def evaluate_model(model, dataset):
    predictions = model.predict(dataset, verbose=1)
    return predictions.flatten()

def save_trained_model(model, file_name='lstm_model.keras'):
    save_model(model, file_name)
    logging.info("Model saved as %s", file_name)

def main():
    check_gpu_availability()
    
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        if df is not None:
            explore_data(df)
            X, y, scaler = preprocess_data(df)
            
            input_shape = (X.shape[1], X.shape[2])
            
            # Adjust batch size for multi-GPU training
            batch_size = 32 * strategy.num_replicas_in_sync
            
            # Split data into training, validation, and testing sets
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            logging.info("y_train shape: %s", y_train.shape)
            logging.info("y_val shape: %s", y_val.shape)
            logging.info("y_test shape: %s", y_test.shape)
            
            # Create datasets
            train_dataset = create_dataset(X_train, y_train, batch_size)
            val_dataset = create_eval_dataset(X_val, y_val, batch_size)
            test_dataset = create_eval_dataset(X_test, y_test, batch_size)
            
            # Calculate steps per epoch
            steps_per_epoch = len(X_train) // batch_size
            
            # Build and compile the model within the strategy scope
            with strategy.scope():
                model = build_model(input_shape)
            
            # Train the model
            train_model(model, train_dataset, val_dataset, steps_per_epoch=steps_per_epoch)
            
            # Evaluate the model
            predictions = evaluate_model(model, test_dataset)
            
            # Flatten y_test to match the predictions shape
            y_test_flat = y_test.flatten()
            
            # Calculate evaluation metrics
            mse = mean_squared_error(y_test_flat, predictions)
            mae = mean_absolute_error(y_test_flat, predictions)
            mape = mean_absolute_percentage_error(y_test_flat, predictions)
            r2 = r2_score(y_test_flat, predictions)
            logging.info("Mean Squared Error: %f", mse)
            logging.info("Mean Absolute Error: %f", mae)
            logging.info("Mean Absolute Percentage Error: %f", mape)
            logging.info("R-squared: %f", r2)
            
            # Plot error distribution
            errors = y_test_flat - predictions
            plt.figure(figsize=(12, 6))
            sns.histplot(errors, bins=50, kde=True)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.show()
            
            # Plot actual vs. predicted values
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_flat, label='Actual Price')
            plt.plot(predictions, label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title('Actual vs. Predicted Prices')
            plt.legend()
            plt.show()
            
            # Save the trained model in native Keras format
            save_trained_model(model)

if __name__ == "__main__":
    main()
