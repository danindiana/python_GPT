import requests
import sys

def get_candles(symbol, interval):
    # Gemini sandbox API endpoint for candles
    url = f"https://api.sandbox.gemini.com/v2/candles/{symbol}/{interval}"
    
    try:
        # Send a GET request to the endpoint
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        candles_data = response.json()
        
        # Print the candle data
        print(f"Candle data for {symbol} with {interval} interval:")
        for candle in candles_data:
            timestamp, open_price, high_price, low_price, close_price, volume = candle
            print(f"Timestamp: {timestamp}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}")
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        print(f"JSON decoding error occurred: {json_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

def main():
    # Default values
    symbol = "btcusd"
    interval = "15m"
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    if len(sys.argv) > 2:
        interval = sys.argv[2]
    
    get_candles(symbol, interval)

if __name__ == "__main__":
    main()
