import os
import pandas as pd

def list_csv_files(directory):
    """List all CSV files in the specified directory."""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return csv_files

def display_csv_files(csv_files):
    """Display the list of CSV files with a numbered list."""
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")

def filter_candles(csv_filename):
    """Filter out rows where all price values are 1.0 and volume is 0.0."""
    df = pd.read_csv(csv_filename)
    filtered_df = df[(df['open'] != 1.0) | (df['high'] != 1.0) | (df['low'] != 1.0) | (df['close'] != 1.0) | (df['volume'] != 0.0)]
    return filtered_df

def main():
    # Get the current directory
    current_directory = os.getcwd()
    
    # List all CSV files in the current directory
    csv_files = list_csv_files(current_directory)
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    # Display the list of CSV files
    display_csv_files(csv_files)
    
    # Prompt the user to select a file
    try:
        choice = int(input("Enter the number of the file you wish to clean: "))
        if choice < 1 or choice > len(csv_files):
            print("Invalid choice. Please enter a valid number.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    # Get the selected file
    selected_file = csv_files[choice - 1]
    
    # Filter the candles
    filtered_df = filter_candles(selected_file)
    
    # Save the filtered DataFrame to a new CSV file
    filtered_csv_filename = f"filtered_{selected_file}"
    filtered_df.to_csv(filtered_csv_filename, index=False)
    
    print(f"Filtered candle data saved to {filtered_csv_filename}")

if __name__ == "__main__":
    main()
