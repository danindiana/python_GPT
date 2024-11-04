To organize the given instructions into a single program or several smaller programs, we can break down the tasks as follows:

1. **Environment Setup**
2. **Data Loading and Preprocessing**
3. **Model Building**
4. **Training the Model**
5. **Evaluation and Prediction**
6. **Saving the Model**

Here is how you can structure these tasks into separate Python files for better organization and reusability.

### 1. Environment Setup
**File Name: `setup.py`**

```python
import tensorflow as tf

def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if __name__ == "__main__":
    check_gpu_availability()
```

### 2. Data Loading and Preprocessing
**File Name: `data_processing.py`**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

if __name__ == "__main__":
    df = load_data('your_data.csv')
    X, y, scaler = preprocess_data(df)
    print("Data preprocessing complete.")
```

### 3. Model Building
**File Name: `model_building.py`**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape)
    print("Model built.")
```

### 4. Training the Model
**File Name: `training.py`**

```python
from tensorflow.keras.callbacks import TensorBoard

def train_model(model, X, y, batch_size=32, epochs=10):
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

if __name__ == "__main__":
    from data_processing import load_data, preprocess_data
    from model_building import build_model
    
    df = load_data('your_data.csv')
    X, y, scaler = preprocess_data(df)
    
    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape)
    
    train_model(model, X, y)
    print("Model training complete.")
```

### 5. Evaluation and Prediction
**File Name: `evaluation.py`**

```python
from tensorflow.keras.models import load_model

def evaluate_and_predict(model, X_test, scaler):
    predictions = model.predict(X_test)
    # Inverse transform the predictions to get actual values
    predictions = scaler.inverse_transform(predictions)
    return predictions

if __name__ == "__main__":
    from data_processing import load_data, preprocess_data
    from training import train_model
    
    df = load_data('your_data.csv')
    X, y, scaler = preprocess_data(df)
    
    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape)
    
    train_model(model, X, y)
    
    # Assuming you have a test set X_test
    predictions = evaluate_and_predict(model, X_test, scaler)
    print("Model evaluation and prediction complete.")
```

### 6. Saving the Model
**File Name: `save_model.py`**

```python
from tensorflow.keras.models import save_model

def save_trained_model(model, file_name='lstm_model.h5'):
    save_model(model, file_name)

if __name__ == "__main__":
    from training import train_model
    
    df = load_data('your_data.csv')
    X, y, scaler = preprocess_data(df)
    
    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape)
    
    train_model(model, X, y)
    
    save_trained_model(model)
    print("Model saved.")
```

### Summary
- **`setup.py`**: Checks GPU availability.
- **`data_processing.py`**: Loads and preprocesses data.
- **`model_building.py`**: Builds the LSTM model.
- **`training.py`**: Trains the model.
- **`evaluation.py`**: Evaluates and predicts using the trained model.
- **`save_model.py`**: Saves the trained model.

This structure allows each task to be modular, making it easier to maintain and extend in the future.
