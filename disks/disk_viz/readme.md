![Screenshot_2025-02-04_15-03-30](https://github.com/user-attachments/assets/8ff604e2-74c7-4129-9fa8-a6c8d9a7f736)


A self‐contained Python script that “drills down” into a directory tree and displays file and directory sizes as a scrolling, interactive, ASCII “bar‐chart” view in your terminal. (It’s inspired by tools like **ncdu** but written from scratch for illustration.) You can save this as, for example, **diskvis.py** and run it on Ubuntu 22.04 with Python 3.

> **Note:** Scanning very large directories recursively can be slow. In a production tool you might want to add threading, caching, or call out to “du” for faster performance.

---

```python
#!/usr/bin/env python3
"""
Disk File Size Visualization Tool
==================================

This tool scans a given directory recursively, computes the sizes of files and
directories, and displays an interactive view in the terminal using curses.
You can navigate using the arrow keys (or j/k), press Enter to “drill down”
into a directory, Backspace to go up, and q to quit.

Usage:
    ./diskvis.py [path]

If no path is provided, the current directory (".") is used.
"""

import os
import curses
import argparse
import locale

# Ensure proper locale settings for unicode block characters
locale.setlocale(locale.LC_ALL, '')

# Global cache for computed sizes (to speed up repeated lookups)
size_cache = {}

def format_size(num_bytes):
    """Return human‐readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"

def get_size(path):
    """
    Recursively compute the size of a file or directory.
    Uses a global cache to avoid duplicate work.
    """
    if path in size_cache:
        return size_cache[path]

    try:
        if os.path.isfile(path) or os.path.islink(path):
            size = os.path.getsize(path)
        elif os.path.isdir(path):
            total = 0
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        entry_path = entry.path
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        elif entry.is_dir(follow_symlinks=False):
                            total += get_size(entry_path)
                    except Exception:
                        # Skip files that cannot be read
                        pass
            size = total
        else:
            size = 0
    except Exception:
        size = 0

    size_cache[path] = size
    return size

def get_entries(path):
    """
    Returns a sorted list of entries in the given directory.
    Each entry is a tuple: (name, full_path, is_directory, size).
    Sorted in descending order by size.
    """
    entries = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    entry_path = entry.path
                    if entry.is_dir(follow_symlinks=False):
                        size = get_size(entry_path)
                    else:
                        size = entry.stat(follow_symlinks=False).st_size
                    entries.append((entry.name, entry_path, entry.is_dir(follow_symlinks=False), size))
                except Exception:
                    # Skip entries that cause errors (e.g. permission issues)
                    pass
    except Exception:
        pass

    # Sort by size (largest first)
    return sorted(entries, key=lambda x: x[3], reverse=True)

def draw_window(stdscr, current_path, entries, selected_idx, start_idx):
    """
    Draws the header and the list of entries with an ASCII bar showing relative size.
    """
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    # Header with current path and instructions
    header = f"Disk Usage Viewer - {current_path} (q: quit, Enter: open, Backspace: up)"
    stdscr.addstr(0, 0, header[:width-1])

    if not entries:
        stdscr.addstr(2, 0, "No entries found or permission denied.")
        stdscr.refresh()
        return

    # Determine the maximum size among entries to scale the bar display.
    max_size = max(e[3] for e in entries) if entries else 1
    # Reserve space for text (name and size)
    bar_max_width = width - 40 if width > 40 else 10

    # Calculate how many lines can be displayed (leave 2 header lines)
    display_lines = height - 2

    for i in range(display_lines):
        idx = start_idx + i
        if idx >= len(entries):
            break

        name, path, is_dir, size = entries[idx]
        size_str = format_size(size)
        # Build an ASCII bar proportional to the entry's size.
        bar_length = int((size / max_size) * bar_max_width) if max_size > 0 else 0
        bar = "█" * bar_length
        line = f"{name:<30.30} {size_str:>10} {bar}"

        # Highlight the currently selected line.
        if idx == selected_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(i+2, 0, line[:width-1])
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(i+2, 0, line[:width-1])
    stdscr.refresh()

def disk_usage_viewer(stdscr, start_path):
    """
    The main loop: display the current directory’s entries and allow the user
    to navigate up and down and “drill down” into directories.
    """
    curses.curs_set(0)  # Hide the cursor
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)

    # Start at the given directory (converted to an absolute path)
    current_path = os.path.abspath(start_path)
    # History stack: (previous_path, selected_index, start_index)
    path_history = []
    selected_idx = 0
    start_idx = 0

    entries = get_entries(current_path)

    while True:
        draw_window(stdscr, current_path, entries, selected_idx, start_idx)
        key = stdscr.getch()

        if key in [ord('q'), ord('Q')]:
            break  # Quit the program

        # Navigate down the list.
        elif key in [curses.KEY_DOWN, ord('j')]:
            if selected_idx < len(entries) - 1:
                selected_idx += 1
                height, _ = stdscr.getmaxyx()
                if selected_idx - start_idx >= height - 2:
                    start_idx += 1

        # Navigate up the list.
        elif key in [curses.KEY_UP, ord('k')]:
            if selected_idx > 0:
                selected_idx -= 1
                if selected_idx < start_idx:
                    start_idx = selected_idx

        # Enter key: if a directory is selected, drill down.
        elif key in [curses.KEY_ENTER, 10, 13]:
            if entries:
                name, path, is_dir, size = entries[selected_idx]
                if is_dir:
                    # Save current state for going back
                    path_history.append((current_path, selected_idx, start_idx))
                    current_path = path
                    selected_idx = 0
                    start_idx = 0
                    entries = get_entries(current_path)

        # Backspace: go up one directory if possible.
        elif key in [curses.KEY_BACKSPACE, 127, 8]:
            parent = os.path.dirname(current_path)
            if parent and parent != current_path:
                if path_history:
                    # Return to the previous directory and state.
                    current_path, selected_idx, start_idx = path_history.pop()
                else:
                    current_path = parent
                    selected_idx = 0
                    start_idx = 0
                entries = get_entries(current_path)

        # Ignore any other key presses.

def main():
    parser = argparse.ArgumentParser(
        description="Disk File Size Visualization Tool (CLI-based)"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to scan (default: current directory)"
    )
    args = parser.parse_args()
    curses.wrapper(disk_usage_viewer, args.path)

if __name__ == "__main__":
    main()
```

---

### How It Works

1. **Scanning and Size Computation:**  
   The functions `get_size()` and `get_entries()` recursively compute file/directory sizes. Results are cached in the global `size_cache` dictionary so that if you navigate back to a directory the work isn’t repeated.

2. **Formatting the Output:**  
   The `draw_window()` function uses the terminal size to print a header and then lists entries (with names, human‐readable sizes, and an ASCII “bar” whose length is proportional to the file/directory size).

3. **Interactive Navigation:**  
   The `disk_usage_viewer()` function is the main curses loop. You can move the selection with the arrow keys (or j/k), press Enter to “open” a directory (drilling down), and use Backspace to return to the parent directory. Press **q** to exit.

4. **Running the Tool:**  
   Save the code as **diskvis.py**, make it executable (`chmod +x diskvis.py`), and run it:
   ```bash
   ./diskvis.py /path/to/scan
   ```
   If no path is provided, it will use the current directory.

This example script should work out-of-the-box on Ubuntu 22.04 (or any modern Linux system with Python 3 installed). Enjoy exploring your disk usage in an ergonomic CLI view!
