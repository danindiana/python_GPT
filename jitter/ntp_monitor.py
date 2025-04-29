#!/usr/bin/env python3
import subprocess
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LOG_FILE = "ntp_offset_log.csv"
POLL_INTERVAL = 60  # seconds

timestamps, offsets, jitters = [], [], []

def get_ntp_values():
    try:
        result = subprocess.run(
            ["ntpq", "-c", "rv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout
        offset = jitter = None
        for item in output.split(','):
            if 'offset=' in item:
                offset = float(item.split('=')[1])
            elif 'sys_jitter=' in item:
                jitter = float(item.split('=')[1])
        return datetime.now(), offset, jitter
    except Exception as e:
        print(f"[Error] Failed to parse ntpq output: {e}")
        return datetime.now(), None, None

def init_csv():
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "offset_ms", "sys_jitter_ms"])
    except FileExistsError:
        pass  # Already exists

def log_to_csv(timestamp, offset, jitter):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp.isoformat(), offset, jitter])

def update_graph(i):
    timestamp, offset, jitter = get_ntp_values()
    if offset is not None and jitter is not None:
        timestamps.append(timestamp)
        offsets.append(offset)
        jitters.append(jitter)
        log_to_csv(timestamp, offset, jitter)

        ax1.clear()
        ax2.clear()
        ax1.plot(timestamps, offsets, label="Offset (ms)", marker='o')
        ax2.plot(timestamps, jitters, label="Sys Jitter (ms)", color='orange', marker='x')
        ax1.legend()
        ax2.legend()
        ax1.set_ylabel("Offset (ms)")
        ax2.set_ylabel("Sys Jitter (ms)")
        ax2.set_xlabel("Time")
        plt.tight_layout()

if __name__ == "__main__":
    init_csv()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ani = animation.FuncAnimation(fig, update_graph, interval=POLL_INTERVAL * 1000)
    print(f"Monitoring NTP offset every {POLL_INTERVAL} seconds. Ctrl+C to stop.")
    plt.show()
