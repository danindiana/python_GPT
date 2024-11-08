import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def list_csv_files():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    return files

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.dropna()
    if data['timestamp'].dtype == 'O':
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except ValueError as e:
            print(f"Error converting timestamp: {e}")
            exit()
    else:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    return data

def moving_averages(data):
    data['price_rolling_mean'] = data['price'].rolling(window=20).mean()
    data['price_rolling_std'] = data['price'].rolling(window=20).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['price'], label='Price')
    plt.plot(data['timestamp'], data['price_rolling_mean'], label='20-period Moving Average')
    plt.fill_between(data['timestamp'], 
                     data['price_rolling_mean'] - data['price_rolling_std'], 
                     data['price_rolling_mean'] + data['price_rolling_std'], 
                     color='b', alpha=0.1, label='Rolling Std Dev')
    plt.title('Price with Moving Average and Standard Deviation')
    plt.legend()
    plt.savefig("moving_averages.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def trade_type_comparison(data):
    buy_data = data[data['type'] == 'buy']
    sell_data = data[data['type'] == 'sell']
    
    plt.figure(figsize=(12, 6))
    plt.plot(buy_data['timestamp'], buy_data['price'], label='Buy Price')
    plt.plot(sell_data['timestamp'], sell_data['price'], label='Sell Price')
    plt.title('Buy vs Sell Price over Time')
    plt.legend()
    plt.savefig("trade_type_comparison.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def volume_over_time(data):
    data['amount_cumsum'] = data['amount'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], data['amount_cumsum'], label='Cumulative Volume')
    plt.title('Cumulative Volume over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Cumulative Amount')
    plt.legend()
    plt.savefig("volume_over_time.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def trade_count_per_interval(data, interval='1h'):
    data.set_index('timestamp', inplace=True)
    trade_counts = data.resample(interval).size()
    data.reset_index(inplace=True)
    
    plt.figure(figsize=(12, 6))
    ax = trade_counts.plot(title='Trade Counts per Interval')
    plt.xlabel('Timestamp')
    plt.ylabel('Trade Count')

    if trade_counts.index.min() != trade_counts.index.max():
        ax.set_xlim(trade_counts.index.min(), trade_counts.index.max())
    
    plt.savefig("trade_count_per_interval.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def lag_plot(data, lag=1):
    pd.plotting.lag_plot(data['price'], lag=lag)
    plt.title(f'Lag Plot with Lag={lag}')
    plt.savefig("lag_plot.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def seasonal_decomposition(data, freq=20):
    from statsmodels.tsa.seasonal import seasonal_decompose
    data.set_index('timestamp', inplace=True)
    decomposition = seasonal_decompose(data['price'], model='additive', period=freq)
    
    plt.figure(figsize=(12, 8))
    decomposition.plot()
    plt.savefig("seasonal_decomposition.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()
    data.reset_index(inplace=True)

def price_vs_amount_scatter(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['price'], data['amount'], alpha=0.5)
    plt.title('Price vs Amount Scatter Plot')
    plt.xlabel('Price')
    plt.ylabel('Amount')
    plt.savefig("price_vs_amount_scatter.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def box_plot_price_by_time(data, interval='hour'):
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    
    if interval == 'hour':
        sns.boxplot(x='hour', y='price', data=data)
        plt.title('Box Plot of Price by Hour of Day')
    elif interval == 'day':
        sns.boxplot(x='day', y='price', data=data)
        plt.title('Box Plot of Price by Day')
    
    plt.savefig("box_plot_price_by_time.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()

def price_change_heatmap(data, interval='1h'):
    data['price_change'] = data['price'].diff()
    data.set_index('timestamp', inplace=True)
    resampled_data = data['price_change'].resample(interval).mean().reset_index()

    resampled_data['hour'] = resampled_data['timestamp'].dt.hour
    resampled_data['day'] = resampled_data['timestamp'].dt.date
    pivot_table = resampled_data.pivot(index="day", columns="hour", values="price_change")

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='coolwarm', center=0, cbar_kws={'label': 'Price Change'})
    plt.title('Price Change Heatmap')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day')
    plt.savefig("price_change_heatmap.png")
    
    if mpl.get_backend() != 'agg':
        plt.show()
    data.reset_index(inplace=True)

def visualize_selected_option(data, option):
    if option == 1:
        moving_averages(data)
    elif option == 2:
        trade_type_comparison(data)
    elif option == 3:
        volume_over_time(data)
    elif option == 4:
        trade_count_per_interval(data)
    elif option == 5:
        lag_plot(data)
    elif option == 6:
        seasonal_decomposition(data)
    elif option == 7:
        price_vs_amount_scatter(data)
    elif option == 8:
        box_plot_price_by_time(data)
    elif option == 9:
        price_change_heatmap(data)
    else:
        print("Invalid option selected.")

if __name__ == "__main__":
    csv_files = list_csv_files()
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        exit()

    print("Select a CSV file to process:")
    for idx, file in enumerate(csv_files):
        print(f"{idx + 1}. {file}")

    file_index = int(input("Enter the number of the file: ")) - 1

    if file_index < 0 or file_index >= len(csv_files):
        print("Invalid selection.")
        exit()

    selected_file = csv_files[file_index]
    print(f"Processing file: {selected_file}")

    data = load_data(selected_file)
    clean_data = preprocess_data(data)

    print("\nVisualization Options:")
    print("1. Moving Averages and Rolling Statistics")
    print("2. Price and Amount by Trade Type")
    print("3. Volume over Time")
    print("4. Trade Count per Time Interval")
    print("5. Lag Plot for Autocorrelation")
    print("6. Seasonal Decomposition of Time Series")
    print("7. Scatter Plot of Amount vs. Price")
    print("8. Box Plot for Price by Time Intervals")
    print("9. Heatmap of Price Changes Over Time")

    option = int(input("Select a visualization option (1-9): "))
    visualize_selected_option(clean_data, option)
