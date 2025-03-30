import json

# filepath: /home/jeb/.cache/lm-studio/conversations/1743364966948.conversation.json
input_file = "/home/jeb/.cache/lm-studio/conversations/1743364966948.conversation.json"
output_file = "/home/jeb/converted_text.txt"

def extract_text(data):
    """Recursively extract all text values from the JSON structure."""
    text_content = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "text" and isinstance(value, str):
                text_content.append(value)
            else:
                text_content.extend(extract_text(value))
    elif isinstance(data, list):
        for item in data:
            text_content.extend(extract_text(item))
    return text_content

# Read the JSON file
with open(input_file, "r") as file:
    json_data = json.load(file)

# Extract all text content
all_text = extract_text(json_data)

# Write the extracted text to a file
with open(output_file, "w") as file:
    file.write("\n\n".join(all_text))

print(f"Text content has been extracted and saved to {output_file}")
