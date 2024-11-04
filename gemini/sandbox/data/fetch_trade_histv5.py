import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

def get_trade_history(symbol, start_time=None, end_time=None):
    url = f"https://api.sandbox.gemini.com/v1/trades/{symbol}"
    
    params = {}
    if start_time:
        params['timestamp'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['limit_trades'] = 500
    
    max_retries = 5
    retry_count = 0
    backoff_time = 5  # Initial backoff time in seconds
    
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 502:
                print(f"Received 502 Bad Gateway. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                retry_count += 1
                backoff_time *= 2  # Double the backoff time for the next retry
            else:
                print(f"HTTP error occurred: {http_err}")
                break
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
            break
        except ValueError as json_err:
            print(f"JSON decoding error occurred: {json_err}")
            break
        except Exception as err:
            print(f"An unexpected error occurred: {err}")
            break
    
    return None

def get_time_period():
    print("Select the time period for fetching trade history:")
    print("1. Last 1 day")
    print("2. Last 7 days")
    print("3. Last 30 days")
    print("4. Custom time period")
    
    choice = input("Enter the number of your choice: ")
    
    if choice == '1':
        return timedelta(days=1)
    elif choice == '2':
        return timedelta(days=7)
    elif choice == '3':
        return timedelta(days=30)
    elif choice == '4':
        start_date = input("Enter the start date (YYYY-MM-DD): ")
        end_date = input("Enter the end date (YYYY-MM-DD): ")
        try:
            start_time = datetime.strptime(start_date, '%Y-%m-%d')
            end_time = datetime.strptime(end_date, '%Y-%m-%d')
            return end_time - start_time
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return None
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")
        return None

def main():
    symbol = "btcusd"
    
    # Get the time period from the user
    time_period = get_time_period()
    if not time_period:
        return
    
    # Define the time range for data retrieval
    end_time = datetime.now()
    start_time = end_time - time_period
    
    print(f"Fetching trade history for {symbol} from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_trades = []
    current_start_time = start_time
    total_trades_fetched = 0
    
    try:
        while current_start_time < end_time:
            print(f"Fetching trades from {current_start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            trades_data = get_trade_history(symbol, start_time=current_start_time, end_time=end_time)
            
            if not trades_data:
                print("No more trades to fetch.")
                break
            
            all_trades.extend(trades_data)
            num_trades = len(trades_data)
            total_trades_fetched += num_trades
            print(f"Fetched {num_trades} trades. Total trades fetched: {total_trades_fetched}")
            
            # Update the current start time to a time after the last trade fetched
            current_start_time = datetime.fromtimestamp(trades_data[-1]['timestampms'] / 1000) + timedelta(seconds=1)
            
            # Ensure the current start time does not exceed the end time
            if current_start_time >= end_time:
                print("Reached the end of the specified time range.")
                break
            
            # Check if there are more trades to fetch
            if len(trades_data) < 500:
                print("Reached the end of available trades.")
                break
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Saving fetched data...")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades, columns=['timestamp', 'timestampms', 'tid', 'price', 'amount', 'exchange', 'type'])
    
    # Save to CSV
    csv_filename = f"historical_trade_data_{symbol}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

if __name__ == "__main__":
    main()
