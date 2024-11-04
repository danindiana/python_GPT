![Figure_22](https://github.com/user-attachments/assets/a2aec410-917b-4574-a7dc-a808eba1d3a6)
![Figure_21](https://github.com/user-attachments/assets/5627b877-a69d-483c-849e-6aad7ef5c036)
![Figure_23](https://github.com/user-attachments/assets/9dc4bde7-6975-418f-a0c4-3d89cf88d8ff)
![Figure_24](https://github.com/user-attachments/assets/93dc3fc2-edba-40c2-8256-5a18747d9ef6)

import os  # Used for interacting with the operating system (e.g., setting environment variables)
import pandas as pd  # Used for data manipulation and analysis (creates DataFrames)
import numpy as np  # Used for numerical operations (arrays, matrices, etc.)
import matplotlib.pyplot as plt  # Used for data visualization (creating plots and charts)
from sklearn.preprocessing import StandardScaler  # Used for feature scaling (standardization)
from sklearn.model_selection import train_test_split  # Used for splitting data into training and testing sets
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # Used for evaluating model performance
import tensorflow as tf  # Open-source machine learning framework
from tensorflow.keras.models import Model, save_model  # Used for defining and saving Keras models
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Flatten  # Used for building neural network layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # Used for monitoring training and implementing early stopping

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sets the environment variable to suppress TensorFlow informational messages

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')  # Gets a list of available GPUs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # Allows GPU memory growth to avoid allocating all memory at once

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()  # Creates a MirroredStrategy object for distributing training across multiple GPUs

def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # Prints the number of available GPUs

def load_data(file_path):
    df = pd.read_csv(file_path)  # Reads data from a CSV file into a pandas DataFrame
    return df.sort_values(by='timestampms')  # Sorts the DataFrame by the 'timestampms' column

def explore_data(df):
    # Summary statistics
    print(df.describe())  # Prints descriptive statistics of the DataFrame

    # Visualizations
    plt.figure(figsize=(12, 6))  # Creates a new figure with specified size
    plt.plot(df['timestampms'], df['price'])  # Plots 'price' against 'timestampms'
    plt.xlabel('Timestamp')  # Sets the x-axis label
    plt.ylabel('Price')  # Sets the y-axis label
    plt.title('Price Over Time')  # Sets the plot title
    plt.show()  # Displays the plot

def preprocess_data(df, seq_length=5, window_size=3):
    # Add a moving average feature
    df['price_ma'] = df['price'].rolling(window=window_size).mean()  # Calculates the moving average of 'price'

    # Add volatility feature
    df['price_volatility'] = df['price'].rolling(window=window_size).std()  # Calculates the rolling standard deviation of 'price'

    # Add momentum indicator
    df['price_momentum'] = df['price'].diff(window_size)  # Calculates the price difference over a given window

    # Add Relative Strength Index (RSI)
    delta = df['price'].diff()  # Calculates the price difference
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()  # Calculates the average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()  # Calculates the average loss
    rs = gain / loss  # Calculates the relative strength (RS)
    df['rsi'] = 100 - (100 / (1 + rs))  # Calculates the RSI

    # Check for NaN values
    if df.isnull().values.any():  # Checks if there are any missing values in the DataFrame
        print("Warning: NaN values found in the data. Dropping NaN rows.")  # Prints a warning message
        df = df.dropna()  # Drops rows with missing values

    features = df[['price', 'amount', 'price_ma', 'price_volatility', 'price_momentum', 'rsi']].values  # Selects the features for the model
    target = df['price'].values  # Selects the target variable

    scaler = StandardScaler()  # Creates a StandardScaler object for feature scaling
    features = scaler.fit_transform(features)  # Scales the features using standardization

    def create_sequences(data, seq_length):  # Function to create sequences of data for the LSTM model
        X, y = [], []  # Initializes empty lists for input sequences (X) and target values (y)
        for i in range(len(data) - seq_length):  # Iterates through the data to create sequences
            X.append(data[i:i+seq_length])  # Appends a sequence of length 'seq_length' to X
            y.append(data[i+seq_length - 1][0])  # Appends the last value of the sequence to y
        return np.array(X), np.array(y)  # Returns the sequences and target values as NumPy arrays

    X, y = create_sequences(features, seq_length)  # Creates the sequences from the features
    print("Generated X shape:", X.shape)  # Prints the shape of the input sequences
    print("Generated y shape:", y.shape)  # Prints the shape of the target values
    return X, y, scaler  # Returns the input sequences, target values, and the scaler object

def list_csv_files(directory):  # Function to list CSV files in a directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]  # Gets a list of CSV files in the directory
    if not csv_files:  # Checks if there are any CSV files
        print("No CSV files found in the directory.")  # Prints a message if no CSV files are found
        return None  # Returns None if no CSV files are found

    print("Available CSV files:")  # Prints a header
    for i, file_name in enumerate(csv_files):  # Iterates through the list of CSV files
        print(f"{i+1}. {file_name}")  # Prints the file name with a number

    choice = input("Enter the number of the file you want to load: ")  # Prompts the user to select a file
    try:
        choice = int(choice)  # Tries to convert the user input to an integer
        if 1 <= choice <= len(csv_files):  # Checks if the choice is valid
            return os.path.join(directory, csv_files[choice-1])  # Returns the path to the selected file
        else:
            print("Invalid choice. Please enter a valid number.")  # Prints an error message
            return None  # Returns None if the choice is invalid
    except ValueError:
        print("Invalid input. Please enter a number.")  # Prints an error message
        return None  # Returns None if the input is invalid

def build_model(input_shape):  # Function to build the LSTM model
    inputs = Input(shape=input_shape)  # Defines the input layer with the specified shape
    lstm1 = LSTM(250, return_sequences=True, dropout=0.2)(inputs)  # Adds an LSTM layer with 250 units and dropout
    lstm2 = LSTM(250, return_sequences=True, dropout=0.2)(lstm1)  # Adds another LSTM layer
    attention = Attention()([lstm2, lstm2])  # Adds an Attention layer
    flatten = Flatten()(attention)  # Flattens the output of the Attention layer
    dense1 = Dense(200, activation='relu')(flatten)  # Adds a Dense layer with 200 units and ReLU activation
    dropout1 = Dropout(0.3)(dense1)  # Adds a Dropout layer with a rate of 0.3
    dense2 = Dense(100, activation='relu')(dropout1)  # Adds another Dense layer with 100 units and ReLU activation
    dropout2 = Dropout(0.3)(dense2)  # Adds another Dropout layer
    outputs = Dense(1)(dropout2)  # Adds the output layer with 1 unit

    model = Model(inputs=inputs, outputs=outputs)  # Creates the model
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compiles the model with the Adam optimizer and MSE loss
    return model  # Returns the compiled model

def create_dataset(X, y, batch_size, shuffle_buffer_size=10000):  # Function to create a TensorFlow Dataset
    dataset = tf.data.Dataset
    .from_tensor_slices((X, y))  # Creates a Dataset from the input sequences (X) and target values (y)
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat()  # Shuffles, batches, and repeats the Dataset
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Prefetches data for improved performance
    return dataset  # Returns the created Dataset

def create_eval_dataset(X, y, batch_size):  # Function to create an evaluation Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))  # Creates a Dataset from the input sequences (X) and target values (y)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Batches and prefetches the Dataset
    return dataset  # Returns the created Dataset

def train_model(model, train_dataset, val_dataset, epochs=60, steps_per_epoch=None):  # Function to train the model
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)  # Creates a TensorBoard callback for visualization
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Creates an EarlyStopping callback

    model.fit(  # Trains the model
        train_dataset,  # Training Dataset
        epochs=epochs,  # Number of training epochs
        steps_per_epoch=steps_per_epoch,  # Number of steps per epoch
        validation_data=val_dataset,  # Validation Dataset
        callbacks=[tensorboard_callback, early_stopping],  # List of callbacks
        verbose=2  # Verbosity level
    )

def evaluate_model(model, dataset):  # Function to evaluate the model
    predictions = model.predict(dataset, verbose=1)  # Gets predictions from the model
    return predictions.flatten()  # Returns the flattened predictions

def save_trained_model(model, file_name='lstm_model.keras'):  # Function to save the trained model
    save_model(model, file_name)  # Saves the model to a file
    print(f"Model saved as {file_name}")  # Prints a confirmation message

def main():  # Main function
    check_gpu_availability()  # Checks and prints the number of available GPUs

    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)  # Lists the CSV files in the directory and gets the selected file path

    if file_path:  # Checks if a file path was selected
        df = load_data(file_path)  # Loads the data from the selected file
        explore_data(df)  # Explores the data (prints statistics and shows a plot)
        X, y, scaler = preprocess_data(df)  # Preprocesses the data

        input_shape = (X.shape[1], X.shape[2])  # Gets the input shape for the model

        # Adjust batch size for multi-GPU training
        batch_size = 32 * strategy.num_replicas_in_sync  # Calculates the batch size based on the number of GPUs

        # Split data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # Splits the data into training and temporary sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Splits the temporary set into validation and testing sets

        print("y_train shape:", y_train.shape)  # Prints the shape of the training target values
        print("y_val shape:", y_val.shape)  # Prints the shape of the validation target values
        print("y_test shape:", y_test.shape)  # Prints the shape of the testing target values

        # Create datasets
        train_dataset = create_dataset(X_train, y_train, batch_size)  # Creates the training Dataset
        val_dataset = create_eval_dataset(X_val, y_val, batch_size)  # Creates the validation Dataset
        test_dataset = create_eval_dataset(X_test, y_test, batch_size)  # Creates the testing Dataset

        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // batch_size  # Calculates the number of steps per epoch

        # Build and compile the model within the strategy scope
        with strategy.scope():  # Enters the distribution strategy scope
            model = build_model(input_shape)  # Builds the model

        # Train the model
        train_model(model, train_dataset, val_dataset, steps_per_epoch=steps_per_epoch)  # Trains the model

        # Evaluate the model
        predictions = evaluate_model(model, test_dataset)  # Evaluates the model and gets predictions

        # Flatten y_test to match the predictions shape
        y_test_flat = y_test.flatten()  # Flattens the testing target values

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_flat, predictions)  # Calculates the Mean Squared Error
        mae = mean_absolute_error(y_test_flat, predictions)  # Calculates the Mean Absolute Error
        mape = mean_absolute_percentage_error(y_test_flat, predictions)  # Calculates the Mean Absolute Percentage Error
        print(f"Mean Squared Error: {mse}")  # Prints the MSE
        print(f"Mean Absolute Error: {mae}")  # Prints the MAE
        print(f"Mean Absolute Percentage Error: {mape}")  # Prints the MAPE

        # Plot error distribution
        errors = y_test_flat - predictions  # Calculates the prediction errors
        plt.figure(figsize=(12, 6))  # Creates a new figure
        plt.hist(errors, bins=50)  # Plots a histogram of the errors
        plt.xlabel('Prediction Error')  # Sets the x-axis label
        plt.ylabel('Frequency')  # Sets the y-axis label
        plt.title('Distribution of Prediction Errors')  # Sets the plot title
        plt.show()  # Displays the plot

        # Plot actual vs. predicted values
        plt.figure(figsize=(12, 6))  # Creates a new figure
        plt.plot(y_test_flat, label='Actual Price')  # Plots the actual prices
        plt.plot(predictions, label='Predicted Price')  # Plots the predicted prices
        plt.xlabel('Time')  # Sets the x-axis label
        plt.ylabel('Price')  # Sets the y-axis label
        plt.title('Actual vs. Predicted Prices')  # Sets the plot title
        plt.legend()  # Shows the legend
        plt.show()  # Displays the plot

        # Save the trained model in native Keras format
        save_trained_model(model)  # Saves the trained model

if __name__ == "__main__":  # Checks if the script is being run as the main program
    main()  # Calls the main function
