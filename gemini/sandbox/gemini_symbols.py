import requests
import json
import logging

# Base URL for Gemini API endpoints
base_url = "https://api.gemini.com/v1"

# Configure basic logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_symbols():
    """Retrieve all available trading symbols."""
    url = f"{base_url}/symbols"
    logging.info(f"Fetching symbols from {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to retrieve symbols: {response.text}")
            raise Exception("Failed to retrieve symbols")
    except Exception as e:
        logging.error(f"Exception occurred while fetching symbols: {e}")
        raise

def create_symbol_lookup_table():
    """Create a dictionary mapping each symbol to its details."""
    try:
        symbols = get_symbols()
        symbol_details_dict = {}
        
        for symbol in symbols:
            url = f"{base_url}/symbols/details/{symbol}"
            logging.info(f"Fetching details for {symbol} from {url}")
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    symbol_details_dict[symbol] = response.json()
                else:
                    logging.error(f"Failed to retrieve details for {symbol}: {response.text}")
            except Exception as e:
                logging.error(f"Exception occurred while fetching details for {symbol}: {e}")
        
        return symbol_details_dict
    except Exception as e:
        logging.error(f"Exception occurred in create_symbol_lookup_table: {e}")
        raise

def save_symbol_data(symbol_data, filename):
    """Save the symbol data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(symbol_data, f)
        logging.info(f"Saved symbol data to {filename}")
    except Exception as e:
        logging.error(f"Failed to save symbol data: {e}")

def load_symbol_data(filename):
    """Load the symbol data from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"No saved symbol data found at {filename}")
        return None
    except Exception as e:
        logging.error(f"Failed to load symbol data: {e}")
        raise

def get_symbol_details(symbol, symbol_lookup_table):
    """Retrieve detailed information about a specific symbol."""
    if symbol in symbol_lookup_table:
        return symbol_lookup_table[symbol]
    else:
        raise KeyError(f"No details found for symbol: {symbol}")

# Example usage
if __name__ == "__main__":
    try:
        logging.info("Starting script execution")
        
        # Check if we have a saved file to load
        filename = "gemini_symbols.json"
        symbol_lookup_table = load_symbol_data(filename)
        
        if not symbol_lookup_table:
            # If no data is found, fetch and save it
            symbol_lookup_table = create_symbol_lookup_table()
            save_symbol_data(symbol_lookup_table, filename)
        
        # Retrieve and print BTCUSD details as an example
        btcusd_details = get_symbol_details("btcusd", symbol_lookup_table)
        logging.info(f"Details for BTCUSD: {btcusd_details}")
    
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
