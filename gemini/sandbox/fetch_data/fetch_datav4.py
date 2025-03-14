import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base URL for Gemini API endpoints
base_url = "https://api.gemini.com/v1"

# Configure basic logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_symbols():
    """Retrieve all available trading symbols."""
    url = f"{base_url}/symbols"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            symbols = response.json()
            logger.info(f"Retrieved symbols: {symbols}")
            return symbols
        else:
            logger.error(f"Failed to retrieve symbols: {response.text}")
            raise Exception("Failed to retrieve symbols")
    except Exception as e:
        logger.error(f"Exception occurred while fetching symbols: {e}")
        sys.exit(1)

def fetch_symbol_details(symbol):
    """Fetch details for a single symbol."""
    url = f"{base_url}/symbols/details/{symbol}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return symbol, response.json()
        else:
            logger.error(f"Failed to retrieve details for {symbol}: {response.text}")
            return symbol, None
    except Exception as e:
        logger.error(f"Exception occurred while fetching details for {symbol}: {e}")
        return symbol, None

def create_symbol_lookup_table():
    """Create a dictionary mapping each symbol to its details."""
    try:
        symbols = get_symbols()
        symbol_details_dict = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_details, symbol): symbol for symbol in symbols}
            for future in as_completed(future_to_symbol):
                symbol, details = future.result()
                if details:
                    symbol_details_dict[symbol] = details
        
        return symbol_details_dict
    except Exception as e:
        logger.error(f"Exception occurred in create_symbol_lookup_table: {e}")
        sys.exit(1)

def save_symbol_data(symbol_data, filename):
    """Save the symbol data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(symbol_data, f)
        logger.info(f"Saved symbol data to {filename}")
    except Exception as e:
        logger.error(f"Failed to save symbol data: {e}")

def load_symbol_data(filename):
    """Load the symbol data from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"No saved symbol data found at {filename}")
        return None
    except Exception as e:
        logger.error(f"Failed to load symbol data: {e}")
        sys.exit(1)

def get_symbol_details(symbol, symbol_lookup_table):
    """Retrieve detailed information about a specific symbol."""
    if symbol in symbol_lookup_table:
        return symbol_lookup_table[symbol]
    else:
        raise KeyError(f"No details found for symbol: {symbol}")

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
                logger.info(f"Received 502 Bad Gateway. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                retry_count += 1
                backoff_time *= 2  # Double the backoff time for the next retry
            else:
                logger.error(f"HTTP error occurred: {http_err}")
                break
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}")
            break
        except ValueError as json_err:
            logger.error(f"JSON decoding error occurred: {json_err}")
            break
        except Exception as err:
            logger.error(f"An unexpected error occurred: {err}")
            break
    
    return None

def get_time_period():
    logger.info("Select the time period for fetching trade history:")
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
            logger.error("Invalid date format. Please use YYYY-MM-DD.")
            return None
    else:
        logger.error("Invalid choice. Please enter a number between 1 and 4.")
        return None

def main():
    # Load or create symbol lookup table
    filename = "gemini_symbols.json"
    symbol_lookup_table = load_symbol_data(filename)
    
    if not symbol_lookup_table:
        symbol_lookup_table = create_symbol_lookup_table()
        save_symbol_data(symbol_lookup_table, filename)

    # Prompt user to select a symbol
    logger.info("Available symbols:")
    for i, symbol in enumerate(symbol_lookup_table.keys(), start=1):
        print(f"{i}. {symbol}")
    
    choice = input("Enter the number of your chosen symbol: ")
    try:
        symbol_index = int(choice) - 1
        if 0 <= symbol_index < len(symbol_lookup_table):
            selected_symbol = list(symbol_lookup_table.keys())[symbol_index]
        else:
            logger.error("Invalid selection. Exiting.")
            return
    except ValueError:
        logger.error("Invalid input. Please enter a number.")
        return
    
    # Get the local system's time/date and convert it to Unix time
    local_time = datetime.now()
    local_unix_time = int(local_time.timestamp())
    logger.info(f"Local system time/date: {local_time.strftime('%Y-%m-%d %H:%M:%S')} (Unix time: {local_unix_time})")
    
    # Get the time period from the user
    time_period = get_time_period()
    if not time_period:
        return
    
    # Define the time range for data retrieval
    end_time = datetime.now()
    start_time = end_time - time_period
    
    logger.info(f"Fetching trade history for {selected_symbol} from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_trades = []
    current_start_time = start_time
    total_trades_fetched = 0
    
    try:
        while current_start_time < end_time:
            logger.info(f"Fetching trades from {current_start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            trades_data = get_trade_history(selected_symbol, start_time=current_start_time, end_time=end_time)
            
            if not trades_data:
                logger.info("No more trades to fetch.")
                break
            
            all_trades.extend(trades_data)
            num_trades = len(trades_data)
            total_trades_fetched += num_trades
            logger.info(f"Fetched {num_trades} trades. Total trades fetched: {total_trades_fetched}")
            
            # Update the current start time to a time after the last trade fetched
            current_start_time = datetime.fromtimestamp(trades_data[-1]['timestampms'] / 1000) + timedelta(seconds=1)
            
            # Ensure the current start time does not exceed the end time
            if current_start_time >= end_time:
                logger.info("Reached the end of the specified time range.")
                break
            
            # Check if there are more trades to fetch
            if len(trades_data) < 500:
                logger.info("Reached the end of available trades.")
                break
    
    except KeyboardInterrupt:
        logger.warning("\nProgram interrupted by user. Saving fetched data...")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades, columns=['timestamp', 'timestampms', 'tid', 'price', 'amount', 'type'])
    
    # Convert all timestamps to Unix time
    df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.fromtimestamp(x / 1000).timestamp()))
    df['timestampms'] = df['timestampms'].apply(lambda x: int(x / 1000))
    
    # Save to CSV
    csv_filename = f"historical_trade_data_{selected_symbol}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"Data saved to {csv_filename}")

if __name__ == "__main__":
    main()
