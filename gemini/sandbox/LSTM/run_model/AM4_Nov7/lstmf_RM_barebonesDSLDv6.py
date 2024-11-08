import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Attention, Concatenate

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
    lstm2 = LSTM(300, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    attention = Attention()([lstm2, lstm2])
    dense1 = Dense(250)(attention)
    dense2 = Dense(200)(dense1)
    dense3 = Dense(150)(dense2)
    outputs = Dense(1)(dense3)  # Single unit for single value prediction

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def preprocess_data(df, seq_length=5, window_size=3):
    # Add features (you can customize these)
    df['price_ma'] = df['price'].rolling(window=window_size).mean()
    df['price_volatility'] = df['price'].rolling(window=window_size).std()
    df['price_momentum'] = df['price'].diff(window_size)
    # ... Add more features as needed

    if df.isnull().values.any():
        print("Warning: NaN values found in the data. Dropping NaN rows.")
        df = df.dropna()

    features = df[['price', 'amount', 'price_ma', 'price_volatility', 'price_momentum']].values  # Define your features
    target = df['price'].values  # Predicting the next price point

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

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.sort_values(by='timestampms') 

def list_csv_files():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the current directory.")
        return None

    print("Available CSV files:")
    for i, file_name in enumerate(csv_files):
        print(f"{i+1}. {file_name}")
    
    while True:
        try:
            choice = int(input("Enter the number of the file you want to load: ")) - 1
            if 0 <= choice < len(csv_files):
                return os.path.join('.', csv_files[choice])
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def save_trained_model(model, file_path):
    model.save(file_path)
    print(f"Model saved to {file_path}")

def main():
    file_path = list_csv_files() 
    if file_path:
        df = load_data(file_path)  
        X, y, scaler = preprocess_data(df)
        input_shape = (X.shape[1], X.shape[2])

        # Determine the number of available GPUs
        num_gpus = len(tf.config.list_physical_devices('GPU')) 

        # Configure TensorFlow to use multiple GPUs
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():  
                model = build_model(input_shape)
                print("Training on", strategy.num_replicas_in_sync, "devices")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
                print(f"Length of X_train: {len(X_train)}")
                print(f"Length of y_train: {len(y_train)}")
                print(f"Length of X_test: {len(X_test)}")
                print(f"Length of y_test: {len(y_test)}")

                history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

        else:
            model = build_model(input_shape)
            print("Training on CPU")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
            print(f"Length of X_train: {len(X_train)}")
            print(f"Length of y_train: {len(y_train)}")
            print(f"Length of X_test: {len(X_test)}")
            print(f"Length of y_test: {len(y_test)}")

            history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate the model (remove if you don't need evaluation)
        predictions = model.predict(X_test)
        predictions = predictions.flatten()[:len(y_test)]  # Ensure predictions have the same length as y_test

        # Check shapes
        print(f"Shape of y_test: {y_test.shape}")
        print(f"Shape of predictions: {predictions.shape}")

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Absolute Percentage Error: {mape}")

        save_trained_model(model, 'lstm_model.keras')

if __name__ == "__main__":
    main()
