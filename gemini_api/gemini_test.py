import google.generativeai as genai
import os
import sys # Import sys to check python path if needed later

print(f"--- Script Starting ---")
print(f"Python executable: {sys.executable}") # See which python is running

# Configure the API key (Best practice: use environment variables)
print(f"Attempting to retrieve GEMINI_API_KEY from environment...")
api_key = os.getenv("GEMINI_API_KEY")

# --- CRUCIAL DEBUG LINE ---
# Let's see exactly what os.getenv returned. Using arrows to clearly show start/end.
print(f"DEBUG: Value from os.getenv('GEMINI_API_KEY'): -->{api_key}<--")
# --------------------------

if not api_key:
    # This condition is TRUE if api_key is None OR an empty string ""
    print(f"ERROR: The 'if not api_key:' check evaluated to TRUE.")
    print(f"ERROR: This means api_key is either None or an empty string.")
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")
else:
    # This will only print if the key is found and is not empty
    print(f"INFO: API key retrieved successfully. Length: {len(api_key)}")

try:
    print("INFO: Configuring genai...")
    genai.configure(api_key=api_key)
    print("INFO: genai configured.")

    # Choose a model
    print("INFO: Selecting model gemini-pro...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or other suitable models
    print("INFO: Model selected.")

    # Send a prompt
    prompt = "Explain what a virtual machine is in simple terms."
    print(f"INFO: Sending prompt: '{prompt}'")
    response = model.generate_content(prompt)
    print("--- Gemini Response ---")
    print(response.text)
    print("---------------------")

except Exception as e:
    print(f"!!! An error occurred during genai processing: {e}")

print("--- Script Finished ---")
