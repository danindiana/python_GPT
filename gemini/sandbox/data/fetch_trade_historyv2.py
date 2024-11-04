import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_trade_history(symbol, start_time=None, end_time=None):
    url = f"https://api.sandbox.gemini.com/v1/trades/{symbol}"
    
    params = {}
    if start_time:
        params['timestamp'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['limit_trades'] = 500
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 502:
                print(f"Received 502 Bad Gateway. Retrying in 5 seconds...")
                time.sleep(5)
                retry_count += 1
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

def main():
    symbol = "btcusd"
    
    # Define the time range for data retrieval
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    all_trades = []
    current_start_time = start_time
    
    while current_start_time < end_time:
        trades_data = get_trade_history(symbol, start_time=current_start_time, end_time=end_time)
        
        if not trades_data:
            break
        
        all_trades.extend(trades_data)
        
        # Update the current start time to the timestamp of the last trade fetched
        current_start_time = datetime.fromtimestamp(trades_data[-1]['timestampms'] / 1000)
        
        # Check if there are more trades to fetch
        if len(trades_data) < 500:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades, columns=['timestamp', 'timestampms', 'tid', 'price', 'amount', 'exchange', 'type'])
    
    # Save to CSV
    csv_filename = f"historical_trade_data_{symbol}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

if __name__ == "__main__":
    main()
