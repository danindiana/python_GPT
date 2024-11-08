```markdown
# Historical Trade Data Visualizer

This project is a Python script for visualizing historical trade data from CSV files. It provides a range of customizable visualizations to help analyze trade data patterns, trends, and suitability for time-series forecasting with models like LSTM.

## Features

- Lists CSV files in the current directory and allows the user to select one to visualize.
- Cleans and preprocesses the selected CSV data, handling both Unix timestamps and date strings.
- Offers various visualization options, including:
  - Moving averages and rolling statistics
  - Price and amount by trade type
  - Volume over time
  - Trade count per interval
  - Lag plot for autocorrelation analysis
  - Seasonal decomposition of time series
  - Scatter plot of amount vs. price
  - Box plot for price by time intervals (hourly or daily)
  - Heatmap of price changes over time (by hour and day)

Each visualization is saved as a `.png` file and displayed interactively (if supported by the environment).

## Setup

### Prerequisites

- Python 3.x
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) for plotting
- [Seaborn](https://seaborn.pydata.org/) for advanced visualizations
- [Statsmodels](https://www.statsmodels.org/) for seasonal decomposition analysis

Install dependencies using `pip`:

```bash
pip install pandas matplotlib seaborn statsmodels
```

### Repository Structure

The repository structure is as follows:

```plaintext
.
├── README.md               # Project documentation
├── visualizer.py           # Main script file for data visualization
└── data/
    └── *.csv               # Place your CSV files here
```

Place your CSV files with historical trade data in the `data/` directory or in the same directory as the `visualizer.py` script.

## Usage

1. **Run the Script**

   Execute the script to start the program:

   ```bash
   python visualizer.py
   ```

2. **Select a CSV File**

   The script will display a list of available CSV files in the current directory. Enter the number corresponding to the file you want to analyze.

3. **Choose a Visualization**

   The program offers several visualization options, each corresponding to a different analysis type. Enter the number associated with the visualization you'd like to generate:

   - **1**: Moving Averages and Rolling Statistics
   - **2**: Price and Amount by Trade Type
   - **3**: Volume over Time
   - **4**: Trade Count per Time Interval
   - **5**: Lag Plot for Autocorrelation
   - **6**: Seasonal Decomposition of Time Series
   - **7**: Scatter Plot of Amount vs. Price
   - **8**: Box Plot for Price by Time Intervals
   - **9**: Heatmap of Price Changes Over Time

4. **Output**

   Each visualization will be saved as a `.png` file in the current directory (e.g., `moving_averages.png`, `volume_over_time.png`). If your environment supports it, the visualization will also display in a pop-up window.

## Example Data Format

The script expects CSV files with the following columns:

- `timestamp`: Unix timestamp or string date (e.g., `2024-11-02`)
- `timestampms`: Millisecond-precision timestamp
- `tid`: Trade ID
- `price`: Price at which the trade occurred
- `amount`: Amount of the asset traded
- `type`: Type of trade (`buy` or `sell`)

### Sample CSV Row

```csv
timestamp,timestampms,tid,price,amount,type
1622548800,1622548800000,123456,35000.5,0.002,buy
```

## Code Overview

The main sections of the `visualizer.py` script:

1. **File Selection**: Lists CSV files and allows the user to select a file.
2. **Data Preprocessing**: Cleans data, drops rows with NaN values, and converts timestamps as needed.
3. **Visualization Options**: Provides a menu for users to choose among nine visualization types.
4. **Plotting and Saving**: Each visualization function saves the plot as a `.png` file and displays it if an interactive backend is available.

## Troubleshooting

- **Non-Interactive Backend**: If you see `FigureCanvasAgg is non-interactive`, it means your environment doesn’t support `plt.show()`. In this case, view the saved `.png` files directly.
- **Identical Low and High xlims Warning**: Occurs if there’s minimal data in the selected interval. The code automatically adjusts for this to prevent plot distortion.

## Contributions

Feel free to fork this repository, make improvements, and submit pull requests. Bug reports and suggestions are also welcome!

## License

This project is licensed under the MIT License.
```

This `README.md` provides clear instructions on setting up, running, and understanding the code, which should be helpful for anyone interacting with the repository.
