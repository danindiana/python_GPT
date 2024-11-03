import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from ratelimit import limits, sleep_and_retry
import time

# Gemini rate limits
RATE_LIMIT_PER_MINUTE = 120
RATE_LIMIT_PER_SECOND = 1

VALID_INTERVALS = ["1m", "5m", "15m", "30m", "1hr", "6hr", "1day"]

@sleep_and_retry
@limits(calls=RATE_LIMIT_PER_MINUTE, period=60)
def get_candles(symbol, interval):
    # Gemini sandbox API endpoint for candles
    url = f"https://api.sandbox.gemini.com/v2/candles/{symbol}/{interval}"
    
    try:
        # Send a GET request to the endpoint
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        candles_data = response.json()
        
        # Convert the list of lists to a pandas DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(candles_data, columns=columns)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save the DataFrame to a CSV file
        csv_filename = f"{symbol}_{interval}_candles.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"Candle data for {symbol} with {interval} interval saved to {csv_filename}")
        
        # Log the number of candles received
        num_candles = len(candles_data)
        print(f"Number of candles received: {num_candles}")
        
        # Calculate expected number of candles based on interval
        if interval == "15m":
            expected_candles = 24 * 4  # 1 candle per 15 minutes in a day
        elif interval == "1d":
            expected_candles = 1  # 1 candle per day
        else:
            expected_candles = None
        
        if expected_candles:
            if num_candles != expected_candles:
                print(f"Warning: Expected {expected_candles} candles but received {num_candles}.")
            else:
                print(f"Number of candles matches expected: {expected_candles}")
    
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 429:
            print("Rate limit exceeded. Retrying after a delay...")
            time.sleep(60)  # Wait for a minute before retrying
            return get_candles(symbol, interval)  # Retry the request
        else:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response content: {response.content}")  # Print the response content for debugging
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        print(f"JSON decoding error occurred: {json_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

def main():
    # Default values
    symbol = "btcusd"
    
    # Ask user for the interval
    interval = input(f"Enter the interval (e.g., {', '.join(VALID_INTERVALS)}): ")
    
    # Validate the interval
    if interval not in VALID_INTERVALS:
        print(f"Invalid interval. Please enter one of the following: {', '.join(VALID_INTERVALS)}")
        return
    
    get_candles(symbol, interval)

if __name__ == "__main__":
    main()
