To implement a Machine Learning Trading Algorithm using the Gemmini API, you'll need to follow a structured approach that includes data collection, preprocessing, model training, and testing. Below is a more detailed and structured example script that covers these steps:

### 1. **Data Collection**

First, you need to collect historical price and volume data from the Gemini API.

```python
import requests

def get_candles(symbol, timeframe='15m'):
    url = f'https://api.gemini.com/v3/candles/{symbol}/{timeframe}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

# Example usage
symbol = 'btcusd'
data = get_candles(symbol)
print(data)
```

### 2. **Data Preprocessing**

Preprocess the collected data to make it suitable for training an ML model.

```python
import pandas as pd

def preprocess_data(data):
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    return df

# Example usage
df = preprocess_data(data)
print(df.head())
```

### 3. **Feature Engineering**

Create features that will be used to train the model.

```python
def create_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# Example usage
df = create_features(df)
print(df.head())
```

### 4. **Model Training**

Train a machine learning model using the preprocessed data.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    X = df[['volatility']]
    y = df['return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Example usage
model, X_test, y_test = train_model(df)
print("Model trained successfully")
```

### 5. **Model Testing**

Test the model by predicting outcomes for the test set.

```python
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return predictions

# Example usage
predictions = test_model(model, X_test, y_test)
print("Predictions:", predictions)
```

### 6. **Evaluation**

Evaluate the model's performance.

```python
from sklearn.metrics import mean_squared_error

def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    return mse

# Example usage
mse = evaluate_model(y_test, predictions)
print("Mean Squared Error:", mse)
```

### 7. **Integration with Trading Strategy**

Integrate the model with a trading strategy to make buy/sell decisions.

```python
def trading_strategy(predictions, df):
    df['signal'] = 0
    df['signal'] = predictions
    df['position'] = df['signal'].diff()
    return df

# Example usage
df = trading_strategy(predictions, df)
print(df.head())
```

### 8. **Execution**

Execute the trading strategy based on the model's predictions.

```python
def execute_trades(df):
    for index, row in df.iterrows():
        if row['position'] > 0:
            print(f"Buy at {row['close']}")
        elif row['position'] < 0:
            print(f"Sell at {row['close']}")

# Example usage
execute_trades(df)
```

### Full Script

Here is the full script combining all the steps:

```python
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Data Collection
def get_candles(symbol, timeframe='15m'):
    url = f'https://api.gemini.com/v3/candles/{symbol}/{timeframe}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

# Data Preprocessing
def preprocess_data(data):
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    return df

# Feature Engineering
def create_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

# Model Training
def train_model(df):
    X = df[['volatility']]
    y = df['return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Model Testing
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return predictions

# Evaluation
def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    return mse

# Trading Strategy
def trading_strategy(predictions, df):
    df['signal'] = 0
    df['signal'] = predictions
    df['position'] = df['signal'].diff()
    return df

# Execution
def execute_trades(df):
    for index, row in df.iterrows():
        if row['position'] > 0:
            print(f"Buy at {row['close']}")
        elif row['position'] < 0:
            print(f"Sell at {row['close']}")

# Main Function
def main():
    symbol = 'btcusd'
    data = get_candles(symbol)
    df = preprocess_data(data)
    df = create_features(df)
    model, X_test, y_test = train_model(df)
    predictions = test_model(model, X_test, y_test)
    mse = evaluate_model(y_test, predictions)
    print("Mean Squared Error:", mse)
    df = trading_strategy(predictions, df)
    execute_trades(df)

if __name__ == "__main__":
    main()
```

This script covers the entire workflow from data collection to executing trades based on the model's predictions. You can further refine and customize it based on your specific requirements and trading strategy.
