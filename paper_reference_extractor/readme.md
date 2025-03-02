Recursive Directory Scanning: The process_directory function now uses os.walk to recursively traverse a directory and find all PDF files.
Output File: The extracted references are written to an output file (extracted_references.txt).
Suspend/Resume:
Signal handlers for SIGUSR1 (suspend) and SIGUSR2 (resume) are implemented.
The suspended flag controls the execution flow.
The program pauses when suspended and resumes when the resume signal is received.
Graceful Shutdown:
Signal handlers for SIGINT and SIGTERM are implemented.
The running flag controls the main loop.
The program finishes processing the current PDF file and then exits.
Signal Handling: The signal handling is more robust.
Error Handling: Includes a try/except block to catch file not found and other exceptions.
Clearer output: the output to the console and to the file are now more consistent.
To use the suspend/resume and shutdown features:

Suspend: Send a SIGUSR1 signal to the process (e.g., kill -SIGUSR1 <pid>).
Resume: Send a SIGUSR2 signal to the process (e.g., kill -SIGUSR2 <pid>).
Shutdown: Press Ctrl+C or send a SIGINT or SIGTERM signal (e.g., kill -SIGTERM <pid>).
To obtain the PID, you can use commands like ps aux | grep your_script_name.py.

JSON Output:
The output_file variable is now set to "extracted_references.json".
The process_pdf function now stores the extracted data in a dictionary.
The process_directory function now stores data in a list of dictionaries.
The results are written to the JSON file using json.dump(all_references, output, indent=4). The indent=4 argument makes the JSON output more readable.
Data Structure: The data is now stored in a list of dictionaries, where each dictionary represents a PDF file and its extracted references, URLs, and DOIs.
This JSON format provides a more structured and easily parsable representation of the extracted references.
