# Simple Google Gemini API Python Script

## Description

This Python script provides a basic example of how to interact with the Google Gemini API using the official `google-generativeai` library. It performs the following actions:

1.  Retrieves a Google Gemini API key securely from environment variables.
2.  Configures the `google-generativeai` client library with the API key.
3.  Selects a specific generative model (e.g., `gemini-1.0-pro`).
4.  Defines a hardcoded text prompt.
5.  Sends the prompt to the selected Gemini model using the `generate_content` method.
6.  Prints the text response received from the API to the console.
7.  Includes basic informational print statements and error handling for the API call.

This script is intended as a starting point for using the Gemini API on a server or development machine.

## Prerequisites

Before you begin, ensure you have the following:

* **A Host Server/Machine:** Any Linux-based server (like Ubuntu, Debian, CentOS) or even macOS/Windows with appropriate terminal access.
* **Python 3:** Check if installed: `python3 --version`. If not, install it using your server's package manager (e.g., `sudo apt update && sudo apt install python3 python3-pip` on Debian/Ubuntu).
* **pip (Python Package Installer):** Usually installed with Python 3. Check with `pip3 --version`.
* **Google Account:** Required to access Google AI Studio.
* **Google Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/).
    * Sign in with your Google Account.
    * Create an API key ("Get API key").
    * **Important:** Copy this key and keep it secure. You'll need it for the configuration step.

## Setup & Installation

Follow these steps to set up the script on your host server:

1.  **Transfer the Script:**
    Copy the `gemini_test.py` script file to a directory on your server (e.g., using `scp`, `git clone` if it's in a repository, or manually creating the file).

2.  **Navigate to Directory:**
    Open a terminal on your server and change to the directory where you placed the script:
    ```bash
    cd /path/to/your/script_directory
    ```

3.  **Create and Activate Virtual Environment (Recommended):**
    Using a virtual environment keeps dependencies isolated for this project.
    ```bash
    # Create the environment (named 'myenv' here)
    python3 -m venv myenv

    # Activate the environment
    # On Linux/macOS:
    source myenv/bin/activate
    # On Windows (Git Bash/WSL): source myenv/Scripts/activate
    # On Windows (CMD): myenv\Scripts\activate.bat
    # On Windows (PowerShell): myenv\Scripts\Activate.ps1
    ```
    *(Your terminal prompt should change to indicate the active environment, e.g., `(myenv) user@host:...`)*

4.  **Install Dependencies:**
    Install the required Google Generative AI library:
    ```bash
    pip install google-generativeai
    ```

## Configuration

The script requires your Gemini API key to authenticate with Google's servers. It reads the key from an environment variable for security reasons (to avoid hardcoding it in the script).

**Set the Environment Variable:**
Before running the script, execute the following command in the **same terminal session** where you plan to run the script. Replace `YOUR_ACTUAL_API_KEY` with the key you obtained from Google AI Studio.

```bash
export GEMINI_API_KEY='YOUR_ACTUAL_API_KEY'

Note: This environment variable is typically only set for the current terminal session. If you close the terminal or start a new session, you will need to run the export command again. For persistent setup, consider adding the export line to your shell's profile script (like ~/.bashrc or ~/.profile), but be mindful of the security implications.

Usage
Once the setup and configuration are complete:

Make sure your virtual environment is active (if you created one).
Make sure the GEMINI_API_KEY environment variable is set in your current terminal session.
Run the script using Python 3:
Bash

python3 gemini_test.py
The script will execute, print some informational messages, and finally output the text response generated by the Gemini API based on the hardcoded prompt.

Script Details
os.getenv("GEMINI_API_KEY"): Securely fetches the API key from the environment.
genai.configure(api_key=api_key): Initializes the library with your credentials.
genai.GenerativeModel('gemini-1.0-pro'): Specifies which Gemini model to use. You can change 'gemini-1.0-pro' to other available models (like 'gemini-1.5-pro-latest'). You might need to use genai.list_models() (see previous conversation steps) to find models available to your key.
prompt = "...": The hardcoded question or instruction sent to the model. You can modify this string to ask different things.
model.generate_content(prompt): Sends the request to the API and waits for the response.
Security Warning
Never hardcode your API key directly into your source code (.py files) or commit it to version control (like Git). Use environment variables as shown, or explore more advanced secrets management solutions (like HashiCorp Vault, cloud provider secret managers) for production applications. Your API key provides access to your Google Cloud resources and quotas; treat it like a password.
