import os
import json

progress_file = "processed_files.log"

def load_progress():
    """Load the list of processed files from the log."""
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()

def save_progress(processed_files):
    """Save the list of processed files to the log."""
    with open(progress_file, "w") as f:
        json.dump(list(processed_files), f)
