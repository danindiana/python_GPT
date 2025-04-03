# Interactive Google Gemini API Client

## Description

This Python script provides an enhanced command-line interface (CLI) for interacting with Google's Gemini AI models. It allows users to dynamically select an available text-generation model, enter a prompt, and view the response rendered directly in the terminal with Markdown formatting.

The script leverages the `google-generativeai` library for API communication and the `rich` library for an improved user interface, including styled output, Markdown rendering, and status indicators.

## Features

* Fetches and lists available Gemini models that support text generation (`generateContent`).
* Provides an interactive numbered menu for the user to select a model.
* Accepts user text prompts interactively.
* Displays Gemini's response with Markdown rendering directly in the terminal.
* Shows a status indicator (spinner) while waiting for the API response.
* Optionally saves the raw Markdown response from Gemini to a file (`gemini_output.md`).
* Uses environment variables for secure handling of the Google Gemini API Key.
* Utilizes the `rich` library for enhanced terminal UI elements (colors, styles, Markdown).

## Prerequisites

Before using this script, ensure you have the following:

* **A Host Machine:** A Linux, macOS, or Windows (with WSL/modern terminal) environment.
* **Python 3:** Version 3.7+ recommended. Check with `python3 --version`. Install if necessary using your system's package manager.
* **pip:** Python's package installer, usually included with Python 3. Check with `pip3 --version`.
* **Google Account:** Required to access Google AI Studio.
* **Google Gemini API Key:**
    * Obtain one from [Google AI Studio](https://aistudio.google.com/).
    * Sign in, navigate to "Get API key", and create one.
    * **Important:** Copy the generated API key and keep it secure.
* **Terminal Support:** A terminal application that supports the formatting features used by `rich` (most modern terminals like Gnome Terminal, Windows Terminal, iTerm2, etc., work well).

## Setup & Installation

1.  **Get the Script:**
    Clone the repository containing the script or download/copy the script file (e.g., save it as `gemini_cli.py`) to a directory on your machine.

2.  **Navigate to Directory:**
    Open your terminal and change to the directory where you saved the script:
    ```bash
    cd /path/to/your/script_directory
    ```

3.  **Create and Activate Virtual Environment (Recommended):**
    Isolating dependencies is good practice.
    ```bash
    # Create a virtual environment named '.venv'
    python3 -m venv .venv

    # Activate it:
    # Linux/macOS (bash/zsh)
    source .venv/bin/activate
    # Windows (Git Bash/WSL)
    # source .venv/Scripts/activate
    # Windows (CMD)
    # .venv\Scripts\activate.bat
    # Windows (PowerShell)
    # .venv\Scripts\Activate.ps1
    ```
    *(Your prompt should change, indicating the active environment)*

4.  **Install Dependencies:**
    Install the necessary Python libraries:
    ```bash
    pip install google-generativeai rich
    ```

## Configuration

The script requires your Gemini API key to authenticate with Google. Set it as an environment variable **in the same terminal session** where you intend to run the script.

```bash
# Replace YOUR_ACTUAL_API_KEY with your real key
export GEMINI_API_KEY='YOUR_ACTUAL_API_KEY'
Note on Persistence: This variable is usually only set for the current session. To make it persistent, you could add the export command to your shell's startup file (e.g., ~/.bashrc, ~/.zshrc, ~/.profile). However, be cautious about storing secrets directly in these files; consider more secure methods for production use.

Usage
Ensure your virtual environment is activated (if you created one).
Ensure the GEMINI_API_KEY environment variable is exported in your current session.
Run the script using Python 3 (replace gemini_cli.py if you named the file differently):
Bash

python3 gemini_cli.py
Follow the Prompts:
The script will list available models. Enter the number corresponding to the model you wish to use.
Once a model is selected, the script will prompt you to enter your text prompt.
View Output:
A status indicator will show while waiting for the API.
The response from Gemini will be displayed in your terminal, rendered as Markdown.
The raw response text will also be saved to gemini_output.md in the same directory.
Example Interaction Flow
$ python3 gemini_cli.py
--- Gemini Interactive Prompt Script (Model Selection) ---
INFO: Retrieving GEMINI_API_KEY from environment...
INFO: API key retrieved successfully.
INFO: Configuring genai...
INFO: genai configured.

INFO: Fetching available models supporting 'generateContent'...
 Listing available models... ✔

Available Models for Text Generation:
  1: Gemini 1.0 Pro (ID: gemini-1.0-pro)
  2: Gemini 1.5 Flash Latest (ID: gemini-1.5-flash-latest)
  3: Gemini 1.5 Pro Latest (ID: gemini-1.5-pro-latest)
  # ... (other models listed) ...

Please select a model number (1-X): 2

INFO: Using model 'gemini-1.5-flash-latest'...
INFO: Model selected and ready.

Now, enter your text prompt.
Please enter your prompt for Gemini: What is the 'rich' library in Python?

INFO: Sending your prompt to Gemini (model: gemini-1.5-flash-latest)...
 Waiting for Gemini response... ✔
INFO: Response received.
INFO: Saving raw response to 'gemini_output.md'...
INFO: Successfully saved raw response.

--- Gemini Response (Rendered via rich) ---
 # (Formatted Markdown output from Gemini appears here, potentially with lists, bolding, etc.)
 # Rich is a Python library for rich text and beautiful formatting in the terminal.
 #
 # * It can print complex data structures in a readable format.
 # * **Supports Markdown rendering.**
 # * Includes progress bars, spinners, tables, and more.
------------------------------------------

--- Script Finished ---
Security Warning
Never hardcode your API key directly into the script file. Treat your API key like a password. Using environment variables is a basic security measure suitable for development. For production or shared environments, consider more robust secrets management solutions (e.g., cloud provider secret managers, HashiCorp Vault).

License
(Optional: Add license information here, e.g., MIT License)
