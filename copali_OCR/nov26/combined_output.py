import os
import json

# Define the directory and output file names
directory = "."  # Change this to your target directory if not the current one
text_output_file = "combined_output.txt"
json_output_file = "combined_output.json"

# Files to ignore
ignored_files = {"processed_files.log"}

# Initialize containers for text and JSON output
all_text_content = []
all_json_content = {}

# Recursively process files in the directory
for root, _, files in os.walk(directory):
    for file in files:
        if file in ignored_files:  # Skip ignored files
            print(f"Skipping {file}")
            continue
        if file.endswith(".py") or file.endswith(".log"):  # Target Python and log files
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    all_text_content.append(f"File: {file}\n{content}\n{'-'*80}\n")
                    all_json_content[file] = content
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Write to the text file
with open(text_output_file, "w") as text_file:
    text_file.write("\n".join(all_text_content))
print(f"All content has been concatenated into {text_output_file}")

# Write to the JSON file
with open(json_output_file, "w") as json_file:
    json.dump(all_json_content, json_file, indent=4)
print(f"All content has been saved in JSON format to {json_output_file}")
