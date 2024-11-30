To selectively **include or exclude documents** based on criteria from the CSV file, we need a program that:

1. **Reads the CSV File**: Load the data containing file names, labels, scores, and any additional metrics (like cosine similarity or combined scores).
2. **Applies Selection Criteria**: Allows the user to define inclusion or exclusion rules, such as:
   - Minimum or maximum scores.
   - Specific labels to include or exclude.
   - Thresholds for combined scores or cosine similarity.
3. **Processes the Selection**: Identifies which documents meet the criteria and optionally:
   - Moves or copies the selected documents to a separate directory.
   - Outputs a new filtered CSV with the selected files.

---

### **Key Features**
- **Dynamic Filtering**:
  - Support user-defined thresholds for scores and similarities.
  - Allow inclusion or exclusion based on labels.
- **Output Options**:
  - Generate a filtered CSV file containing only the selected documents.
  - Optionally copy or move selected files into a specified directory.

---

### **Python Script for Selective Inclusion/Exclusion**

```python
import os
import csv
import shutil

# Define constants
CSV_FILE = "classification_results_with_similarity.csv"
INPUT_DIRECTORY = "./input_files"  # Directory containing original text files
OUTPUT_DIRECTORY = "./filtered_files"  # Directory to save included files
FILTERED_CSV = "filtered_documents.csv"  # Output CSV with selected files

def read_csv(file_path):
    """Read the CSV file and return its content as a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to floats
            row["Score"] = float(row["Score"])
            row["Cosine Similarity"] = float(row["Cosine Similarity"])
            row["Combined Score"] = float(row["Combined Score"]) if "Combined Score" in row else None
            data.append(row)
    return data

def filter_documents(data, label_includes=None, label_excludes=None, min_score=0.0, min_similarity=0.0, min_combined=None):
    """Filter documents based on criteria."""
    filtered = []
    for row in data:
        # Apply inclusion/exclusion criteria
        if label_includes and row["Label"] not in label_includes:
            continue
        if label_excludes and row["Label"] in label_excludes:
            continue
        if row["Score"] < min_score or row["Cosine Similarity"] < min_similarity:
            continue
        if min_combined and row["Combined Score"] is not None and row["Combined Score"] < min_combined:
            continue
        filtered.append(row)
    return filtered

def copy_filtered_files(filtered_data, input_dir, output_dir):
    """Copy filtered files to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for row in filtered_data:
        source_file = os.path.join(input_dir, row["File Name"])
        if os.path.exists(source_file):
            shutil.copy(source_file, output_dir)

def write_filtered_csv(filtered_data, output_file):
    """Write filtered data to a new CSV file."""
    if not filtered_data:
        print("No documents matched the criteria.")
        return
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filtered_data[0].keys())
        writer.writeheader()
        writer.writerows(filtered_data)

def main():
    # Load CSV data
    print("Reading CSV file...")
    data = read_csv(CSV_FILE)

    # Define selection criteria
    print("Applying selection criteria...")
    label_includes = input("Enter labels to include (comma-separated, or leave blank): ").strip().split(",") or None
    label_excludes = input("Enter labels to exclude (comma-separated, or leave blank): ").strip().split(",") or None
    min_score = float(input("Enter minimum classification score (default: 0.0): ") or 0.0)
    min_similarity = float(input("Enter minimum cosine similarity (default: 0.0): ") or 0.0)
    min_combined = float(input("Enter minimum combined score (leave blank if unused): ") or "0")

    # Filter documents
    filtered_data = filter_documents(
        data,
        label_includes=[label.strip() for label in label_includes] if label_includes else None,
        label_excludes=[label.strip() for label in label_excludes] if label_excludes else None,
        min_score=min_score,
        min_similarity=min_similarity,
        min_combined=float(min_combined) if min_combined else None
    )

    # Copy selected files
    print(f"Copying {len(filtered_data)} selected files to {OUTPUT_DIRECTORY}...")
    copy_filtered_files(filtered_data, INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    # Write filtered CSV
    print(f"Saving filtered results to {FILTERED_CSV}...")
    write_filtered_csv(filtered_data, FILTERED_CSV)

    print("Filtering complete!")

if __name__ == "__main__":
    main()
```

---

### **How It Works**

1. **Input Options**:
   - The script prompts for:
     - Labels to include (e.g., "science, technology").
     - Labels to exclude (e.g., "law").
     - Thresholds for classification scores, cosine similarity, and combined scores.

2. **Filtering Logic**:
   - Filters documents based on inclusion/exclusion rules and score thresholds.

3. **Copy Selected Files**:
   - Copies the selected files to a specified output directory (`./filtered_files`).

4. **Generate a New CSV**:
   - Writes a new CSV (`filtered_documents.csv`) containing only the filtered data.

---

### **Example Use**

#### **Input CSV**
```csv
File Name,Label,Score,Cosine Similarity,Combined Score
example1.txt,science,0.8943,0.9123,0.8990
example1.txt,technology,0.7321,0.8412,0.7637
example2.txt,law,0.8712,0.8123,0.8572
example2.txt,biology,0.4512,0.7641,0.5560
```

#### **Input to Script**
```plaintext
Enter labels to include (comma-separated, or leave blank): science, technology
Enter labels to exclude (comma-separated, or leave blank): law
Enter minimum classification score (default: 0.0): 0.7
Enter minimum cosine similarity (default: 0.0): 0.8
Enter minimum combined score (leave blank if unused): 0.85
```

#### **Filtered Output**
**Filtered CSV (`filtered_documents.csv`):**
```csv
File Name,Label,Score,Cosine Similarity,Combined Score
example1.txt,science,0.8943,0.9123,0.8990
```

**Filtered Files Directory (`./filtered_files`):**
- `example1.txt` (copied from the input directory).

---

### **Customizing Further**
1. **Change Output Paths**: Modify the `OUTPUT_DIRECTORY` or `FILTERED_CSV` paths.
2. **Batch Processing**: Extend to process multiple input directories or CSV files.
3. **Logging**: Add detailed logs for better debugging or audit trails.

Let me know if you’d like additional features! 😊

Sure, I can help you with that. Below is the complete Python script that reads a CSV file, applies user-defined selection criteria, processes the selection, and optionally moves or copies the selected documents to a separate directory. It also generates a filtered CSV file containing only the selected documents.

```python
import os
import csv
import shutil

# Define constants
CSV_FILE = "classification_results_with_similarity.csv"
INPUT_DIRECTORY = "./input_files"  # Directory containing original text files
OUTPUT_DIRECTORY = "./filtered_files"  # Directory to save included files
FILTERED_CSV = "filtered_documents.csv"  # Output CSV with selected files

def read_csv(file_path):
    """Read the CSV file and return its content as a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to floats
            row["Score"] = float(row["Score"])
            row["Cosine Similarity"] = float(row["Cosine Similarity"])
            row["Combined Score"] = float(row["Combined Score"]) if "Combined Score" in row else None
            data.append(row)
    return data

def filter_documents(data, label_includes=None, label_excludes=None, min_score=0.0, min_similarity=0.0, min_combined=None):
    """Filter documents based on criteria."""
    filtered = []
    for row in data:
        # Apply inclusion/exclusion criteria
        if label_includes and row["Label"] not in label_includes:
            continue
        if label_excludes and row["Label"] in label_excludes:
            continue
        if row["Score"] < min_score or row["Cosine Similarity"] < min_similarity:
            continue
        if min_combined and row["Combined Score"] is not None and row["Combined Score"] < min_combined:
            continue
        filtered.append(row)
    return filtered

def copy_filtered_files(filtered_data, input_dir, output_dir):
    """Copy filtered files to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for row in filtered_data:
        source_file = os.path.join(input_dir, row["File Name"])
        if os.path.exists(source_file):
            shutil.copy(source_file, output_dir)

def write_filtered_csv(filtered_data, output_file):
    """Write filtered data to a new CSV file."""
    if not filtered_data:
        print("No documents matched the criteria.")
        return
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filtered_data[0].keys())
        writer.writeheader()
        writer.writerows(filtered_data)

def main():
    # Load CSV data
    print("Reading CSV file...")
    data = read_csv(CSV_FILE)

    # Define selection criteria
    print("Applying selection criteria...")
    label_includes = input("Enter labels to include (comma-separated, or leave blank): ").strip().split(",") or None
    label_excludes = input("Enter labels to exclude (comma-separated, or leave blank): ").strip().split(",") or None
    min_score = float(input("Enter minimum classification score (default: 0.0): ") or 0.0)
    min_similarity = float(input("Enter minimum cosine similarity (default: 0.0): ") or 0.0)
    min_combined = input("Enter minimum combined score (leave blank if unused): ").strip() or None
    if min_combined:
        min_combined = float(min_combined)

    # Filter documents
    filtered_data = filter_documents(
        data,
        label_includes=[label.strip() for label in label_includes] if label_includes else None,
        label_excludes=[label.strip() for label in label_excludes] if label_excludes else None,
        min_score=min_score,
        min_similarity=min_similarity,
        min_combined=min_combined
    )

    # Copy selected files
    print(f"Copying {len(filtered_data)} selected files to {OUTPUT_DIRECTORY}...")
    copy_filtered_files(filtered_data, INPUT_DIRECTORY, OUTPUT_DIRECTORY)

    # Write filtered CSV
    print(f"Saving filtered results to {FILTERED_CSV}...")
    write_filtered_csv(filtered_data, FILTERED_CSV)

    print("Filtering complete!")

if __name__ == "__main__":
    main()
```

### How It Works

1. **Input Options**:
   - The script prompts for:
     - Labels to include (e.g., "science, technology").
     - Labels to exclude (e.g., "law").
     - Thresholds for classification scores, cosine similarity, and combined scores.

2. **Filtering Logic**:
   - Filters documents based on inclusion/exclusion rules and score thresholds.

3. **Copy Selected Files**:
   - Copies the selected files to a specified output directory (`./filtered_files`).

4. **Generate a New CSV**:
   - Writes a new CSV (`filtered_documents.csv`) containing only the filtered data.

### Example Use

#### Input CSV
```csv
File Name,Label,Score,Cosine Similarity,Combined Score
example1.txt,science,0.8943,0.9123,0.8990
example1.txt,technology,0.7321,0.8412,0.7637
example2.txt,law,0.8712,0.8123,0.8572
example2.txt,biology,0.4512,0.7641,0.5560
```

#### Input to Script
```plaintext
Enter labels to include (comma-separated, or leave blank): science, technology
Enter labels to exclude (comma-separated, or leave blank): law
Enter minimum classification score (default: 0.0): 0.7
Enter minimum cosine similarity (default: 0.0): 0.8
Enter minimum combined score (leave blank if unused): 0.85
```

#### Filtered Output
**Filtered CSV (`filtered_documents.csv`):**
```csv
File Name,Label,Score,Cosine Similarity,Combined Score
example1.txt,science,0.8943,0.9123,0.8990
```

**Filtered Files Directory (`./filtered_files`):**
- `example1.txt` (copied from the input directory).

### Customizing Further
1. **Change Output Paths**: Modify the `OUTPUT_DIRECTORY` or `FILTERED_CSV` paths.
2. **Batch Processing**: Extend to process multiple input directories or CSV files.
3. **Logging**: Add detailed logs for better debugging or audit trails.
