Creating a backtesting program for the Gemini sandbox involves simulating historical trading conditions using historical price data. Below is a comprehensive guide to help you build a backtesting program in Python. This program will use historical price data to simulate trading decisions based on a simple strategy.

### Step 1: Set Up Your Environment

1. **Install Required Libraries**: Ensure you have the necessary libraries installed in your Python environment.

   ```bash
   pip install pandas numpy matplotlib requests
   ```

### Step 2: Fetch Historical Price Data

You can fetch historical price data from a reliable source or use a CSV file containing historical price data. For this example, we'll assume you have a CSV file with historical price data.

```python
import pandas as pd

# Load historical price data
df = pd.read_csv('btc_price_data.csv')  # Replace with actual data loading
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Display the first few rows of the data
print(df.head())
```

### Step 3: Define Your Trading Strategy

For this example, we'll use a simple moving average crossover strategy. When the short-term moving average crosses above the long-term moving average, we'll buy. When it crosses below, we'll sell.

```python
# Calculate moving averages
short_window = 10
long_window = 50

df['Short_MA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
df['Long_MA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

# Generate trading signals
df['Signal'] = 0
df['Signal'][short_window:] = np.where(df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1, 0)

# Generate trading positions
df['Position'] = df['Signal'].diff()

# Display the DataFrame with signals and positions
print(df.tail())
```

### Step 4: Simulate Trading

Now, let's simulate trading based on the signals generated by our strategy.

```python
import numpy as np

# Initialize variables
initial_capital = 10000.0
positions = pd.DataFrame(index=df.index).fillna(0.0)
portfolio = pd.DataFrame(index=df.index).fillna(0.0)

# Buy and hold 1 unit of BTC when the signal is 1 (buy)
positions['BTC'] = df['Signal']

# Initialize the portfolio with value owned
portfolio['BTC'] = positions['BTC'].multiply(df['Close'], axis=0)

# Store the difference in shares owned
portfolio['Cash'] = initial_capital - (positions['BTC'].diff().multiply(df['Close'], axis=0)).cumsum()

# Total portfolio value
portfolio['Total'] = portfolio['BTC'] + portfolio['Cash']

# Display the portfolio
print(portfolio.tail())
```

### Step 5: Visualize the Results

Finally, let's visualize the results to see how our strategy performed.

```python
import matplotlib.pyplot as plt

# Plot the closing price and moving averages
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['Short_MA'], label=f'{short_window}-day MA')
plt.plot(df['Long_MA'], label=f'{long_window}-day MA')
plt.legend(loc='best')
plt.title('BTCUSD Price and Moving Averages')
plt.show()

# Plot the trading signals
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price')
plt.plot(df.loc[df['Position'] == 1.0].index, df['Close'][df['Position'] == 1.0], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(df.loc[df['Position'] == -1.0].index, df['Close'][df['Position'] == -1.0], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
plt.legend(loc='best')
plt.title('BTCUSD Trading Signals')
plt.show()

# Plot the portfolio value
plt.figure(figsize=(14, 7))
plt.plot(portfolio['Total'], label='Portfolio Value')
plt.legend(loc='best')
plt.title('Portfolio Value Over Time')
plt.show()
```

### Step 6: Evaluate the Strategy

To evaluate the performance of your strategy, you can calculate metrics such as the total return, maximum drawdown, and Sharpe ratio.

```python
# Calculate total return
total_return = (portfolio['Total'][-1] - initial_capital) / initial_capital
print(f'Total Return: {total_return:.2%}')

# Calculate maximum drawdown
drawdown = portfolio['Total'] / portfolio['Total'].cummax() - 1.0
max_drawdown = drawdown.min()
print(f'Maximum Drawdown: {max_drawdown:.2%}')

# Calculate Sharpe ratio (assuming risk-free rate is 0)
returns = portfolio['Total'].pct_change()
sharpe_ratio = returns.mean() / returns.std()
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
```

### Important Considerations

- **Data Quality**: Ensure that your historical price data is accurate and complete.
- **Strategy Complexity**: This example uses a simple moving average crossover strategy. You can experiment with more complex strategies.
- **Risk Management**: Implement risk management techniques to protect against significant losses.
- **Optimization**: Continuously optimize your strategy by adjusting parameters and testing on different datasets.

By following these steps, you can create a backtesting program for the Gemini sandbox. This will allow you to evaluate the performance of your trading strategies in a simulated environment before deploying them in live trading.