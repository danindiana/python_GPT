This is a Python script for time series forecasting using a Long Short-Term Memory (LSTM) recurrent neural network. Here's a breakdown of the code, along with some suggestions for improvement:

**Code Explanation**

* **Libraries:** Imports necessary libraries for data manipulation, preprocessing, model building, and training.
* **Distribution Strategy:**  Uses `tf.distribute.MirroredStrategy` to enable multi-GPU training, potentially speeding up the process.
* **GPU Check:**  Verifies the availability of GPUs.
* **Data Loading and Preprocessing:**
    * `load_data`: Loads data from a CSV file and sorts it by timestamp.
    * `preprocess_data`: 
        * Extracts features ('price', 'amount') and the target variable ('price').
        * Scales features to a range of 0-1 using `MinMaxScaler`.
        * Creates sequences of data with a specified length (`seq_length`) to feed into the LSTM.
* **CSV File Selection:**  Provides a user interface to choose a CSV file from a directory.
* **Model Building:** `build_model` defines the LSTM architecture:
    * An input layer.
    * Two LSTM layers with 50 units each.
    * Two dense layers (fully connected layers) for output processing.
    * The model is compiled with the 'adam' optimizer and 'mean_squared_error' loss function.
* **Dataset Creation:**
    * `create_dataset`:  Creates a TensorFlow `Dataset` from the input data, shuffles it, batches it, and prefetches data for efficient training.
    * `create_eval_dataset`: Creates a similar dataset for evaluation without shuffling.
* **Model Training:** `train_model` trains the model:
    * Uses `TensorBoard` for visualizing training progress.
    * Fits the model to the training data.
* **Model Evaluation:** `evaluate_model` makes predictions and returns them.
* **Model Saving:** `save_trained_model` saves the trained model in Keras format.
* **Main Function:**
    * Orchestrates the entire process: loads data, preprocesses it, builds the model, trains, evaluates, and saves the model.

**Potential Improvements and Considerations**

* **Data Exploration and Feature Engineering:** Before diving into modeling, it's crucial to understand your data. Explore the time series for trends, seasonality, and any anomalies. Consider adding more relevant features or engineered features (e.g., moving averages, lagged variables) that might improve the model's performance.
* **Hyperparameter Tuning:** The script uses default values for many hyperparameters (e.g., number of LSTM units, number of layers, batch size, learning rate). Experiment with these values to find the optimal configuration for your dataset. Consider using techniques like grid search or Bayesian optimization for more efficient hyperparameter tuning.
* **Model Validation:**  The script splits the data into training and testing sets, but it doesn't use a validation set. A validation set helps you monitor the model's performance during training and prevent overfitting.
* **Early Stopping:** Implement early stopping to halt training when the model's performance on the validation set starts to degrade. This helps prevent overfitting.
* **Regularization:** Add regularization techniques (e.g., dropout, L1/L2 regularization) to the LSTM layers to further reduce overfitting.
* **Data Scaling:** While the script uses `MinMaxScaler`, you could explore other scaling methods like `StandardScaler` to see if they improve results.
* **Error Analysis:** Analyze the prediction errors to gain insights into where the model is struggling. This can guide further feature engineering or model adjustments.
* **More Advanced Architectures:**  Consider more complex LSTM architectures (e.g., bidirectional LSTMs, attention mechanisms) or hybrid models that combine LSTMs with other techniques (e.g., CNNs for feature extraction).
* **Deployment:**  Think about how you'll deploy your trained model. You might want to save the model in a format suitable for your deployment environment (e.g., TensorFlow Serving).

**Running the Code**

1.  **Save:** Save the code as `lstm_time_series_forecasting.py`.
2.  **Data:** Place your CSV file (with 'timestampms', 'price', and 'amount' columns) in the same directory as the script or provide the correct directory path.
3.  **Run:** Execute the script from your terminal using `python lstm_time_series_forecasting.py`.

Remember to install the required libraries (TensorFlow, Keras, Pandas, scikit-learn) if you haven't already.




Your code appears to be working well overall, and the output provides the expected metrics and messages. However, here are a few suggestions to optimize and troubleshoot some of the issues you're seeing:

### 1. Suppress Redundant TensorFlow Error Messages
The messages related to the cuFFT, cuDNN, and cuBLAS factories can often be silenced by adding a line at the beginning of your script to suppress TensorFlow warnings:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
```

Setting `'TF_CPP_MIN_LOG_LEVEL'` to `'2'` will reduce some verbosity, while setting it to `'3'` suppresses even more output.

### 2. Handle Out-of-Range Errors Gracefully
The "OUT_OF_RANGE" errors you see during evaluation often indicate that TensorFlow has reached the end of your dataset. To address this, ensure that your evaluation dataset doesn't have `.repeat()` applied. 

You could also wrap the `model.predict` or the entire evaluation block with an error-handling mechanism to capture and ignore this expected error:

```python
try:
    predictions = model.predict(dataset)
except tf.errors.OutOfRangeError:
    print("End of dataset sequence reached during evaluation.")
```

### 3. Enhance GPU Configuration
To make sure TensorFlow is using your GPUs effectively, consider explicitly setting memory growth:

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

This prevents TensorFlow from allocating all GPU memory at once and instead allocates memory as needed, which can avoid certain memory issues, especially if you're running multiple sessions or other GPU-intensive tasks in the background.

### 4. Performance Tracking with TensorBoard
Using TensorBoard to monitor model performance in real-time can be very useful. To view the TensorBoard logs:

1. Run this command in your terminal: `tensorboard --logdir=./logs`.
2. Open the TensorBoard URL it provides (usually `http://localhost:6006`).

### 5. Adjust Training Parameters
To further tune your model performance, consider adjusting the following:
   - **Sequence Length (`seq_length`)**: Test different values to capture the most useful time dependencies in your data.
   - **LSTM Units**: Experiment with the number of units in each LSTM layer. More units often allow the model to capture more complex patterns, though they increase computation.

Let me know if you'd like further help fine-tuning these aspects or adjusting any part of your code.



Thank you for the suggestions. I'll incorporate these optimizations into the script. Here's the updated version of the script with the enhancements:

```python
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    try:
        predictions = model.predict(dataset)
    except tf.errors.OutOfRangeError:
        print("End of dataset sequence reached during evaluation.")
        predictions = None
    return predictions

def save_trained_model(model, file_name='lstm_model.keras'):
    save_model(model, file_name)

def main():
    check_gpu_availability()
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
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
        
        if predictions is not None:
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
```

### Explanation of Changes:

1. **Suppress TensorFlow Warnings**:
   - Added `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'` to suppress TensorFlow warnings.

2. **Handle Out-of-Range Errors Gracefully**:
   - Wrapped the `model.predict` call in a try-except block to handle `tf.errors.OutOfRangeError` gracefully.

3. **Enhance GPU Configuration**:
   - Configured GPU memory growth to prevent TensorFlow from allocating all GPU memory at once.

4. **Performance Tracking with TensorBoard**:
   - Included TensorBoard callback in the `train_model` function to monitor model performance.

5. **Adjust Training Parameters**:
   - No specific changes in this script, but you can experiment with different sequence lengths and LSTM units as needed.

### Running the Script:
- Save the script with the chosen file name (e.g., `lstm_time_series_forecasting.py`).
- Run the script from the command line:
  ```bash
  python lstm_time_series_forecasting.py
  ```

This updated script should provide a more robust and optimized workflow for training and evaluating your LSTM model using multiple GPUs.

It looks like the adjustments were effective in improving the code's behavior, but some repetitive messages about “OUT_OF_RANGE” during evaluation and GPU-related issues persist. Here are some additional strategies to further clean up and optimize the output:

### 1. Suppress Specific TensorFlow Logging (More Granular)
If you're still seeing CUDA factory registration warnings despite the general suppression setting (`TF_CPP_MIN_LOG_LEVEL`), you may want to specifically limit only the relevant logs without suppressing all informative logs. Unfortunately, TensorFlow’s current logging system doesn’t allow for finer control over these specific messages, but updating your TensorFlow version might help reduce these warnings if an update is available.

### 2. Streamline Evaluation to Handle “OUT_OF_RANGE” Exception Gracefully
To eliminate repetitive "OUT_OF_RANGE" logs during evaluation, you can modify the evaluation process slightly. An effective approach is to ensure that your dataset doesn't run indefinitely during evaluation, as the `repeat()` function (when applied to `dataset`) creates a looping effect.

Since you’re already using `create_eval_dataset()` without `.repeat()`, the issue might relate to the end-of-iteration error thrown by the dataset's finite batch. Wrapping the evaluation with a more comprehensive error handling or refactoring may reduce these logs:

```python
def evaluate_model(model, dataset):
    try:
        predictions = model.predict(dataset, verbose=1)
        return predictions
    except tf.errors.OutOfRangeError:
        print("End of dataset sequence reached during evaluation.")
        return None
```

### 3. Improve Logging and Control Verbose Output
During model training (`model.fit()`), the `verbose` argument can help control the output during epochs:
- `verbose=0` (silent),
- `verbose=1` (progress bar),
- `verbose=2` (one line per epoch).

For a more silent mode during training, try adjusting it within `train_model`:

```python
def train_model(model, dataset, epochs=10, steps_per_epoch=None):
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    
    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback],
        verbose=2  # Adjust to 0, 1, or 2 as needed
    )
```

This will condense the output to make it easier to interpret.

### 4. GPU Error Handling & Environment Configuration
For issues related to GPU factory errors, consider updating CUDA and cuDNN drivers or the TensorFlow version if that is possible on your setup. These issues are often related to specific driver and TensorFlow version mismatches.

### 5. Using a Limited Evaluation Callback
If evaluating in a loop or if you frequently call `evaluate_model` during training, you could add an evaluation callback to monitor only selected batches without evaluating the entire dataset every time.

Let me know if you'd like guidance on setting up more sophisticated evaluation routines or tracking model improvements effectively during training!

It looks like you've started to implement some of the suggestions I provided! Here's a breakdown of the changes and further improvements you can make:

**Changes Made**

* **TensorFlow Warnings:** You've added `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'` to suppress less important TensorFlow log messages, which can help clean up the output.
* **GPU Memory Growth:**  You've included code to configure TensorFlow to allocate GPU memory dynamically with `tf.config.experimental.set_memory_growth(gpu, True)`. This is good practice, especially if you have other processes using the GPU.
* **Error Handling in `evaluate_model`:** You've added a `try-except` block to handle the `tf.errors.OutOfRangeError` that can occur when the end of the dataset is reached during evaluation.

**Further Improvements**

Here's how you can incorporate more of the suggestions to enhance your code:

1. **Data Exploration and Feature Engineering:**

   * **Before preprocessing:**
     ```python
     import matplotlib.pyplot as plt

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

         # You can add more plots (e.g., histograms, scatter plots)
         # to explore relationships between features.

     # ... (in main function)
     if file_path:
         df = load_data(file_path)
         explore_data(df)  # Call the exploration function
         X, y, scaler = preprocess_data(df)
         # ... 
     ```

   * **Feature engineering (in `preprocess_data`)**
     ```python
     def preprocess_data(df, seq_length=5):
         # ... (existing code)

         # Example: Add a moving average feature
         df['price_ma'] = df['price'].rolling(window=3).mean()  

         features = df[['price', 'amount', 'price_ma']].values  # Include new feature
         # ... (rest of the preprocessing code)
     ```

2. **Hyperparameter Tuning and Validation Set:**

   * **Create a validation split:**
     ```python
     # ... (in main function)
     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

     # Create datasets
     train_dataset = create_dataset(X_train, y_train, batch_size)
     val_dataset = create_eval_dataset(X_val, y_val, batch_size)  # Validation dataset
     # ...
     ```

   * **Use `validation_data` in `train_model`:**
     ```python
     def train_model(model, train_dataset, val_dataset, epochs=10, steps_per_epoch=None):
         # ...
         model.fit(
             train_dataset,
             epochs=epochs,
             steps_per_epoch=steps_per_epoch,
             validation_data=val_dataset,  # Pass validation data
             callbacks=[tensorboard_callback]
         )
     ```

   * **For more advanced hyperparameter tuning, consider using Keras Tuner or other optimization libraries.**

3. **Early Stopping:**

   ```python
   from tensorflow.keras.callbacks import EarlyStopping

   def train_model(model, train_dataset, val_dataset, epochs=10, steps_per_epoch=None):
       # ...
       early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Stop if val_loss doesn't improve for 3 epochs
       model.fit(
           # ...
           callbacks=[tensorboard_callback, early_stopping]  # Add early stopping callback
       )
   ```

4. **Regularization:**

   ```python
   def build_model(input_shape):
       model = Sequential([
           Input(shape=input_shape),
           LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),  # Add dropout
           LSTM(50, return_sequences=False, dropout=0.2, recurrent_dropout=0.2), # Add dropout
           # ...
       ])
       # ...
   ```

5. **Error Analysis:**

   ```python
   def evaluate_model(model, dataset):
       # ... (your existing code)

       if predictions is not None:
           # ... (calculate MSE)

           # Error analysis
           errors = y_test - predictions
           plt.figure(figsize=(12, 6))
           plt.hist(errors, bins=50)
           plt.xlabel('Prediction Error')
           plt.ylabel('Frequency')
           plt.title('Distribution of Prediction Errors')
           plt.show()
           # You can add more plots or analysis (e.g., scatter plot of 
           # predictions vs. actual values) to understand the errors better.
   ```

**Important Notes:**

* **Experimentation:** Machine learning is iterative. Try different combinations of these improvements and analyze the results to see what works best for your specific dataset.
* **Data:** The quality and characteristics of your data will significantly impact the model's performance.
* **Computational Resources:** Training LSTMs, especially with large datasets, can be computationally intensive. Make sure you have adequate resources.

The output indicates that the model training is progressing without `NaN` values, and the mean squared error (MSE) is being calculated correctly. This is a good sign that the issues with `NaN` values have been resolved.

### Summary of Changes and Improvements:

1. **Data Preprocessing**:
   - Added a check for `NaN` values in the data and dropped rows containing `NaN` values.
   - Added a moving average feature to the data.

2. **Model Building**:
   - Added L2 regularization to the LSTM layers and the dense layer to prevent overfitting.
   - Reduced the learning rate to `0.001` to ensure stable training.

3. **Training and Evaluation**:
   - Added early stopping to prevent overfitting.
   - Used a validation set during training to monitor the model's performance.
   - Added checks for `NaN` values in the predictions.

4. **Error Analysis**:
   - Plotted the distribution of prediction errors to analyze the model's performance.

### Next Steps:

1. **Hyperparameter Tuning**:
   - Experiment with different hyperparameters (e.g., learning rate, number of LSTM units, dropout rates) to further improve the model's performance.

2. **Feature Engineering**:
   - Explore additional features that might improve the model's predictive power (e.g., technical indicators, time-based features).

3. **Model Interpretation**:
   - Use techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to interpret the model's predictions.

4. **Cross-Validation**:
   - Implement cross-validation to ensure that the model's performance is robust and not dependent on the specific train-test split.

5. **Deployment**:
   - Once you are satisfied with the model's performance, consider deploying it in a production environment.

### Final Script:

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Input, LSTM, Dense
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

def preprocess_data(df, seq_length=5):
    # Add a moving average feature
    df['price_ma'] = df['price'].rolling(window=3).mean()
    
    # Check for NaN values
    if df.isnull().values.any():
        print("Warning: NaN values found in the data. Dropping NaN rows.")
        df = df.dropna()
    
    features = df[['price', 'amount', 'price_ma']].values
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
        LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        LSTM(50, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(25, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
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

def train_model(model, train_dataset, val_dataset, epochs=10, steps_per_epoch=None):
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
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
            if y_test.ndim == 2:
                y_test = y_test[:, 0]
            
            if predictions.ndim == 2:
                predictions = predictions[:, 0]
            
            mse = mean_squared_error(y_test, predictions)
            print(f"Mean Squared Error: {mse}")
            
            # Error analysis
            errors = y_test - predictions
            plt.figure(figsize=(12, 6))
            plt.hist(errors, bins=50)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            plt.show()
        
        # Save the trained model in native Keras format
        save_trained_model(model)

if __name__ == "__main__":
    main()
```

### Running the Script:
- Save the script with the chosen file name (e.g., `lstm_time_series_forecasting.py`).
- Run the script from the command line:
  ```bash
  python lstm_time_series_forecasting.py
  ```

This script should now provide a robust and optimized workflow for training and evaluating your LSTM model using multiple GPUs, with enhanced data exploration, hyperparameter tuning, early stopping, regularization, and error analysis.
