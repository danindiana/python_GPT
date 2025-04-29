This Python script monitors and logs NTP (Network Time Protocol) offset and system jitter values over time, displaying them in a real-time graph. Here's a breakdown of what it does:

### Key Features:
1. **NTP Monitoring**:
   - Uses `ntpq -c rv` command to query NTP statistics
   - Extracts offset (time difference from reference) and system jitter (measure of variability)

2. **Data Logging**:
   - Creates a CSV log file (ntp_offset_log.csv) if it doesn't exist
   - Appends new measurements with timestamps in ISO format

3. **Real-time Visualization**:
   - Uses matplotlib to create two stacked plots:
     - Top: NTP offset in milliseconds
     - Bottom: System jitter in milliseconds
   - Updates the graph every 60 seconds (configurable)

4. **Error Handling**:
   - Gracefully handles cases where NTP values can't be parsed

### Usage:
1. Requires:
   - Python 3
   - `ntpq` (part of NTP package)
   - matplotlib, subprocess, csv, datetime modules

2. Run with:
   ```bash
   chmod +x scriptname.py  # if making executable
   ./scriptname.py
   ```
   or
   ```bash
   python3 scriptname.py
   ```

3. The script will:
   - Create/append to the log file
   - Open a window with live-updating graphs
   - Continue running until manually stopped (Ctrl+C)

### Customization:
- Change `POLL_INTERVAL` to adjust how frequently measurements are taken
- Modify `LOG_FILE` to change the output filename
- Adjust `figsize` in `plt.subplots()` to change graph dimensions

The visualization helps identify time synchronization stability issues, showing both the current offset and how much it's fluctuating (jitter).
