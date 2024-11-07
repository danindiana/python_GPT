import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, GRU, Dropout, BatchNormalization, Attention, Concatenate
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.sort_values(by='timestampms')

def explore_data(df):
    # Summary statistics
    print(df.describe())

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestampms'], df['price'])
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.title('Price Over Time')
    plt.show()

def fourier_embeddings(time_index, frequencies):
    embeddings = []
    for freq in frequencies:
        embeddings.append(np.sin(2 * np.pi * freq * time_index))
        embeddings.append(np.cos(2 * np.pi * freq * time_index))
    return np.array(embeddings)

def preprocess_data(df, seq_length=5, window_size=3, frequencies=[1/24, 1/7, 1/30]):
    # Add a moving average feature
    df['price_ma'] = df['price'].rolling(window=window_size).mean()
    
    # Add volatility feature
    df['price_volatility'] = df['price'].rolling(window=window_size).std()
    
    # Add momentum indicator
    df['price_momentum'] = df['price'].diff(window_size)
    
    # Add Relative Strength Index (RSI)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Check for NaN values
    if df.isnull().values.any():
        print("Warning: NaN values found in the data. Dropping NaN rows.")
        df = df.dropna()
    
    # Add Fourier embeddings
    df.loc[:, 'time_index'] = pd.to_datetime(df['timestampms'], unit='ms').index
    df.loc[:, 'fourier_embeddings'] = df['time_index'].apply(lambda x: fourier_embeddings(x, frequencies))

    # Reshape Fourier embeddings to match the dimensions of other features
    fourier_embeddings_reshaped = np.array(df['fourier_embeddings'].tolist())

    # Concatenate Fourier embeddings with the existing features
    features = np.concatenate([df[['price', 'amount', 'price_ma', 'price_volatility', 'price_momentum', 'rsi']].values,
                               fourier_embeddings_reshaped], axis=1)

    # Define target columns
    target = df['price'].values  # Predicting the next price point
    
    # Scale the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])       # Sequence of features
            y.append(target[i+seq_length])       # Single value to predict
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
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    lstm2 = LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.001))(lstm1)
    attention = Attention()([lstm2, lstm2])
    dense1 = Dense(250, kernel_regularizer=tf.keras.regularizers.l2(0.001))(attention)
    dense2 = Dense(200, kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense1)
    dense3 = Dense(150, kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense2)
    outputs = Dense(1)(dense3)  # Single unit for single value prediction
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, early_stopping],
        verbose=2
    )

def evaluate_model(model, dataset):
    try:
        predictions = model.predict(dataset, verbose=1)
        if np.isnan(predictions).any():
            print("Warning: NaN values found in predictions.")
            return None
        return predictions
    except tf.errors.OutOfRangeError:
        print("End of dataset sequence reached during evaluation.")
        return None

def save_trained_model(model, file_name='lstm_model.keras'):
    save_model(model, file_name)

def main():
    check_gpu_availability()
    
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        explore_data(df)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        
        # Adjust batch size for multi-GPU training
        batch_size = 32 * strategy.num_replicas_in_sync
        
        # Split data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
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
        
        if predictions is not None:
            # Ensure that both y_test and predictions have the same shape
            if y_test.ndim > 1:
                y_test = y_test.flatten()
            
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            # Check length consistency
            if len(y_test) != len(predictions):
                # Adjust predictions length to match y_test if needed (e.g., truncate extra entries)
                predictions = predictions[:len(y_test)]
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            print(f"Mean Squared Error: {mse}")
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Absolute Percentage Error: {mape}")
            
            # Error analysis
            errors = y_test - predictions
            plt.figure(figsize=(12, 6))
            plt.hist(errors, bins=50)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.show()
            
            # Plot actual vs. predicted values
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Actual Price')
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
