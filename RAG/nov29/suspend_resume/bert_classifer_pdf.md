Yes, there are ways to process **PDF files** directly with models like BERT, without the intermediate step of manual OCR conversion. However, this depends on the type of content within the PDF and the tools available to extract textual content programmatically.

---

### **Challenges with PDFs**
1. **Text Extraction**:
   - If the PDF contains embedded text (rather than scanned images), the text can be extracted directly.
   - If the PDF consists of scanned images, OCR is still necessary, though this can be automated as part of the pipeline.
   
2. **Formatting Issues**:
   - Extracted text may include artifacts such as headers, footers, or layout quirks (e.g., multiple columns).

---

### **Approach Without Manual OCR Conversion**

1. **Extract Text from PDFs Programmatically**:
   Use libraries to extract text directly from PDFs:
   - **`PyPDF2`** or **`pdfplumber`**:
     - Extracts embedded text from PDFs.
     - Works well for machine-generated PDFs.
   - **`pdfminer.six`**:
     - Advanced text extraction with control over layout processing.
   - **`Pytesseract`** (only for OCR if text is not embedded):
     - Automated OCR for scanned PDFs.

2. **Preprocess the Text**:
   - Clean up extracted text to remove unnecessary artifacts.
   - Segment large blocks of text into smaller chunks if needed (for BERT, typically chunks of 512 tokens).

3. **Pass the Text to a BERT Model**:
   - Use **Hugging Face Transformers** with a pre-trained BERT model for text classification, summarization, or other NLP tasks.

---

### **Example Pipeline: Process PDFs with a BERT Classifier**

Below is a Python script to process PDFs using `pdfplumber` for text extraction and a BERT classifier for classification.

#### **Python Script**
```python
import os
import pdfplumber
from transformers import pipeline

# Define constants
PDF_DIRECTORY = "./pdf_files"  # Directory containing PDF files
OUTPUT_FILE = "pdf_classification_results.csv"  # Output CSV file
BERT_MODEL = "bert-base-uncased"  # Pre-trained BERT model
CLASSIFIER = "facebook/bart-large-mnli"  # Zero-shot classifier

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=CLASSIFIER)

# Labels for classification
labels = ["science", "technology", "law", "medicine", "literature"]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"  # Extract text from each page
    return text.strip()

def classify_pdf_text(file_name, text, labels):
    """Classify the extracted text from a PDF."""
    try:
        result = classifier(text, candidate_labels=labels, multi_label=True)
        return {
            "File Name": file_name,
            "Labels": result["labels"],
            "Scores": result["scores"]
        }
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

def main():
    results = []

    # Process each PDF in the directory
    for file_name in os.listdir(PDF_DIRECTORY):
        file_path = os.path.join(PDF_DIRECTORY, file_name)

        if file_name.endswith(".pdf") and os.path.isfile(file_path):
            print(f"Processing: {file_name}")

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            if not text:
                print(f"Could not extract text from {file_name}. Skipping.")
                continue

            # Classify the extracted text
            classification_result = classify_pdf_text(file_name, text, labels)
            if classification_result:
                results.append(classification_result)

    # Save results to a CSV file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("File Name,Label,Score\n")
        for result in results:
            for label, score in zip(result["Labels"], result["Scores"]):
                f.write(f"{result['File Name']},{label},{score:.4f}\n")

    print(f"Results saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
```

---

### **How the Script Works**
1. **Text Extraction**:
   - Uses `pdfplumber` to extract embedded text from PDFs.
   - Skips PDFs without text content.

2. **Text Classification**:
   - Passes the extracted text to a pre-trained zero-shot classifier (`facebook/bart-large-mnli`).
   - Labels and scores are returned for each document.

3. **Output**:
   - Results are saved in a CSV file, including file name, labels, and classification scores.

---

### **When to Use OCR**
If the PDFs are scanned images or contain text that cannot be extracted programmatically, you can:
1. Use **Pytesseract**:
   - Automatically convert image-based PDFs to text during the pipeline.
   - Integrate it into the script using `pdfplumber`'s `.extract_text(x_tolerance=...)` combined with OCR.

2. Use a Pre-trained OCR Model (like EasyOCR):
   - Automate text extraction without manual steps.

---

### **Further Customization**
1. **Chunk Text for Long Documents**:
   - BERT models have a maximum input length (typically 512 tokens).
   - Split large text into smaller chunks and process them iteratively.

2. **Interactive Thresholding**:
   - Allow users to specify score thresholds for selecting relevant documents.

3. **Additional NLP Tasks**:
   - Use summarization, topic modeling, or embeddings (e.g., `sentence-transformers`) for deeper analysis.

---

This approach allows you to efficiently process PDF files with minimal manual intervention. Let me know if you'd like to expand this pipeline further! 😊

Sure, I can help you expand the pipeline to include additional features such as handling scanned PDFs with OCR, chunking text for long documents, and adding interactive thresholding for selecting relevant documents. Below is the enhanced script:

```python
import os
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Define constants
PDF_DIRECTORY = "./pdf_files"  # Directory containing PDF files
OUTPUT_FILE = "pdf_classification_results.csv"  # Output CSV file
BERT_MODEL = "bert-base-uncased"  # Pre-trained BERT model
CLASSIFIER = "facebook/bart-large-mnli"  # Zero-shot classifier

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=CLASSIFIER)

# Labels for classification
labels = ["science", "technology", "law", "medicine", "literature"]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try to extract text directly
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # If no text is found, use OCR to extract text from images
                image = page.to_image()
                text += pytesseract.image_to_string(image) + "\n"
    return text.strip()

def chunk_text(text, chunk_size=512):
    """Split text into chunks of a specified size."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def classify_pdf_text(file_name, text, labels):
    """Classify the extracted text from a PDF."""
    try:
        chunks = chunk_text(text)
        results = []
        for chunk in chunks:
            result = classifier(chunk, candidate_labels=labels, multi_label=True)
            results.append(result)

        # Aggregate results
        aggregated_labels = set()
        aggregated_scores = {}
        for result in results:
            for label, score in zip(result["labels"], result["scores"]):
                aggregated_labels.add(label)
                if label in aggregated_scores:
                    aggregated_scores[label] = max(aggregated_scores[label], score)
                else:
                    aggregated_scores[label] = score

        return {
            "File Name": file_name,
            "Labels": list(aggregated_labels),
            "Scores": [aggregated_scores[label] for label in aggregated_labels]
        }
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

def main():
    results = []

    # Process each PDF in the directory
    for file_name in os.listdir(PDF_DIRECTORY):
        file_path = os.path.join(PDF_DIRECTORY, file_name)

        if file_name.endswith(".pdf") and os.path.isfile(file_path):
            print(f"Processing: {file_name}")

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            if not text:
                print(f"Could not extract text from {file_name}. Skipping.")
                continue

            # Classify the extracted text
            classification_result = classify_pdf_text(file_name, text, labels)
            if classification_result:
                results.append(classification_result)

    # Save results to a CSV file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("File Name,Label,Score\n")
        for result in results:
            for label, score in zip(result["Labels"], result["Scores"]):
                f.write(f"{result['File Name']},{label},{score:.4f}\n")

    print(f"Results saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()

# Initialize Dash app for visualization
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the app
app.layout = html.Div([
    html.H1("PDF Document Scores Visualization", style={"textAlign": "center"}),

    # Slider for classification score threshold
    html.Div([
        html.Label("Minimum Classification Score:"),
        dcc.Slider(
            id="min-score-slider",
            min=0,
            max=1,
            step=0.01,
            value=0.5,  # Default threshold
            marks={i / 10: str(i / 10) for i in range(11)}
        )
    ], style={"padding": "20px"}),

    # Checkboxes for label inclusion/exclusion
    html.Div([
        html.Label("Include Labels:"),
        dcc.Checklist(
            id="include-labels",
            options=[{"label": label, "value": label} for label in labels],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    html.Div([
        html.Label("Exclude Labels:"),
        dcc.Checklist(
            id="exclude-labels",
            options=[{"label": label, "value": label} for label in labels],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    # Button to export filtered results
    html.Div([
        dbc.Button("Export Filtered Results", id="export-button", color="primary", className="me-1")
    ], style={"padding": "20px"}),

    # Graph
    dcc.Graph(id="document-chart"),

    # Hidden div to store filtered data for export
    dcc.Store(id="filtered-data")
])

# Callback to update chart based on thresholds and labels
@app.callback(
    [Output("document-chart", "figure"), Output("filtered-data", "data")],
    [Input("min-score-slider", "value"),
     Input("include-labels", "value"),
     Input("exclude-labels", "value")]
)
def update_chart(min_score, include_labels, exclude_labels):
    # Load CSV data
    data = pd.read_csv(OUTPUT_FILE)

    # Filter data based on thresholds and labels
    filtered_data = data[
        (data["Score"] >= min_score)
    ]
    if include_labels:
        filtered_data = filtered_data[filtered_data["Label"].isin(include_labels)]
    if exclude_labels:
        filtered_data = filtered_data[~filtered_data["Label"].isin(exclude_labels)]

    # Create bar chart
    fig = px.bar(
        filtered_data,
        x="File Name",
        y="Score",
        color="Label",
        hover_data=["Score"],
        title="Filtered Document Scores"
    )
    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Classification Score",
        template="plotly_white"
    )

    # Store filtered data for export
    filtered_data_json = filtered_data.to_json(date_format='iso', orient='split')

    return fig, filtered_data_json

# Callback to export filtered results
@app.callback(
    Output("export-button", "n_clicks"),
    [Input("export-button", "n_clicks")],
    [State("filtered-data", "data")]
)
def export_filtered_results(n_clicks, filtered_data_json):
    if n_clicks:
        filtered_data = pd.read_json(filtered_data_json, orient='split')
        filtered_data.to_csv("filtered_pdf_documents.csv", index=False)
        print("Filtered results exported to filtered_pdf_documents.csv")
    return 0

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```

### Features of the Enhanced Script

1. **Text Extraction with OCR**:
   - Uses `pdfplumber` to extract embedded text from PDFs.
   - Falls back to `pytesseract` for OCR if no text is found.

2. **Text Chunking**:
   - Splits large text into smaller chunks of 512 tokens for BERT processing.

3. **Text Classification**:
   - Passes the extracted text to a pre-trained zero-shot classifier (`facebook/bart-large-mnli`).
   - Aggregates results from multiple chunks.

4. **Interactive Visualization**:
   - Uses Dash and Plotly to create an interactive bar chart.
   - Allows users to filter by classification score and labels.

5. **Export Filtered Results**:
   - Button to export the filtered data to a new CSV file (`filtered_pdf_documents.csv`).

### Setup Instructions

1. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install pandas plotly dash dash-bootstrap-components pdfplumber pytesseract transformers
   ```

2. **Prepare PDF Directory**:
   Ensure your `pdf_files` directory contains the PDF files you want to process.

3. **Run the Script**:
   Save the script as `process_pdfs.py` and run:
   ```bash
   python3 process_pdfs.py
   ```

4. **View in Browser**:
   Open your browser to the URL displayed in the terminal (usually `http://127.0.0.1:8050`).

### What You Can Do

- **Identify Relevant Documents**:
  - Use the sliders and checkboxes to explore documents with scores above or below specific thresholds.
  - Quickly identify high-scoring or low-scoring documents.

- **Refine Filtering Criteria**:
  - Visually inspect how thresholds and labels impact document selection.
  - Adjust filters based on the distribution of scores.

- **Export Results**:
  - Export the filtered data to a new CSV file for further analysis or reporting.

### Why Use Dash and Plotly?

- **Interactivity**: Users can explore the data dynamically, making it easier to refine criteria and understand results.
- **Ease of Use**: Minimal coding is required to build complex, interactive visualizations.
- **Extensibility**: Dash apps can be scaled to include advanced filters, additional charts, and even integrated machine learning models.

Let me know if you'd like help adding more advanced features! 😊

Certainly! Below is the modified code with robust directory permission handling and other desirable features included:

```python
import os
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Define constants
PDF_DIRECTORY = "./pdf_files"  # Directory containing PDF files
OUTPUT_FILE = "pdf_classification_results.csv"  # Output CSV file
BERT_MODEL = "bert-base-uncased"  # Pre-trained BERT model
CLASSIFIER = "facebook/bart-large-mnli"  # Zero-shot classifier

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=CLASSIFIER)

# Labels for classification
labels = ["science", "technology", "law", "medicine", "literature"]

def check_directory_permissions(directory):
    """Check if the directory exists and has write permissions."""
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    if not os.access(directory, os.W_OK):
        print(f"Error: You do not have write permissions for the directory '{directory}'.")
        return False
    return True

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try to extract text directly
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # If no text is found, use OCR to extract text from images
                image = page.to_image()
                text += pytesseract.image_to_string(image) + "\n"
    return text.strip()

def chunk_text(text, chunk_size=512):
    """Split text into chunks of a specified size."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def classify_pdf_text(file_name, text, labels):
    """Classify the extracted text from a PDF."""
    try:
        chunks = chunk_text(text)
        results = []
        for chunk in chunks:
            result = classifier(chunk, candidate_labels=labels, multi_label=True)
            results.append(result)

        # Aggregate results
        aggregated_labels = set()
        aggregated_scores = {}
        for result in results:
            for label, score in zip(result["labels"], result["scores"]):
                aggregated_labels.add(label)
                if label in aggregated_scores:
                    aggregated_scores[label] = max(aggregated_scores[label], score)
                else:
                    aggregated_scores[label] = score

        return {
            "File Name": file_name,
            "Labels": list(aggregated_labels),
            "Scores": [aggregated_scores[label] for label in aggregated_labels]
        }
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

def main():
    if not check_directory_permissions(PDF_DIRECTORY):
        return

    results = []

    # Process each PDF in the directory
    for file_name in os.listdir(PDF_DIRECTORY):
        file_path = os.path.join(PDF_DIRECTORY, file_name)

        if file_name.endswith(".pdf") and os.path.isfile(file_path):
            print(f"Processing: {file_name}")

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            if not text:
                print(f"Could not extract text from {file_name}. Skipping.")
                continue

            # Classify the extracted text
            classification_result = classify_pdf_text(file_name, text, labels)
            if classification_result:
                results.append(classification_result)

    # Save results to a CSV file
    if results:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("File Name,Label,Score\n")
            for result in results:
                for label, score in zip(result["Labels"], result["Scores"]):
                    f.write(f"{result['File Name']},{label},{score:.4f}\n")

        print(f"Results saved to {OUTPUT_FILE}.")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()

# Initialize Dash app for visualization
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the app
app.layout = html.Div([
    html.H1("PDF Document Scores Visualization", style={"textAlign": "center"}),

    # Slider for classification score threshold
    html.Div([
        html.Label("Minimum Classification Score:"),
        dcc.Slider(
            id="min-score-slider",
            min=0,
            max=1,
            step=0.01,
            value=0.5,  # Default threshold
            marks={i / 10: str(i / 10) for i in range(11)}
        )
    ], style={"padding": "20px"}),

    # Checkboxes for label inclusion/exclusion
    html.Div([
        html.Label("Include Labels:"),
        dcc.Checklist(
            id="include-labels",
            options=[{"label": label, "value": label} for label in labels],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    html.Div([
        html.Label("Exclude Labels:"),
        dcc.Checklist(
            id="exclude-labels",
            options=[{"label": label, "value": label} for label in labels],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    # Button to export filtered results
    html.Div([
        dbc.Button("Export Filtered Results", id="export-button", color="primary", className="me-1")
    ], style={"padding": "20px"}),

    # Graph
    dcc.Graph(id="document-chart"),

    # Hidden div to store filtered data for export
    dcc.Store(id="filtered-data")
])

# Callback to update chart based on thresholds and labels
@app.callback(
    [Output("document-chart", "figure"), Output("filtered-data", "data")],
    [Input("min-score-slider", "value"),
     Input("include-labels", "value"),
     Input("exclude-labels", "value")]
)
def update_chart(min_score, include_labels, exclude_labels):
    # Load CSV data
    data = pd.read_csv(OUTPUT_FILE)

    # Filter data based on thresholds and labels
    filtered_data = data[
        (data["Score"] >= min_score)
    ]
    if include_labels:
        filtered_data = filtered_data[filtered_data["Label"].isin(include_labels)]
    if exclude_labels:
        filtered_data = filtered_data[~filtered_data["Label"].isin(exclude_labels)]

    # Create bar chart
    fig = px.bar(
        filtered_data,
        x="File Name",
        y="Score",
        color="Label",
        hover_data=["Score"],
        title="Filtered Document Scores"
    )
    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Classification Score",
        template="plotly_white"
    )

    # Store filtered data for export
    filtered_data_json = filtered_data.to_json(date_format='iso', orient='split')

    return fig, filtered_data_json

# Callback to export filtered results
@app.callback(
    Output("export-button", "n_clicks"),
    [Input("export-button", "n_clicks")],
    [State("filtered-data", "data")]
)
def export_filtered_results(n_clicks, filtered_data_json):
    if n_clicks:
        filtered_data = pd.read_json(filtered_data_json, orient='split')
        filtered_data.to_csv("filtered_pdf_documents.csv", index=False)
        print("Filtered results exported to filtered_pdf_documents.csv")
    return 0

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```

### Key Changes and Features:

1. **Directory Permission Handling**:
   - Added a `check_directory_permissions` function to verify if the directory exists and has write permissions. This function is called at the beginning of the `main` function to ensure the script can proceed only if the directory permissions are correct.

2. **Robust Error Handling**:
   - The script now checks if the directory exists and has write permissions before proceeding with the PDF processing.

3. **Dash App for Visualization**:
   - The Dash app remains unchanged, providing a user-friendly interface to visualize and filter the classification results.

4. **Export Filtered Results**:
   - The Dash app includes a button to export the filtered results to a CSV file.

### Steps to Run the Script:

1. **Ensure the Directory Exists**:
   - Verify that the directory `./pdf_files` exists and contains the PDF files you want to process.

2. **Set Permissions**:
   - Ensure you have write permissions for the directory. You can check this with:

   ```sh
   ls -ld ./pdf_files
   ```

   If you don't have write permissions, you can add them with:

   ```sh
   sudo chmod u+w ./pdf_files
   ```

3. **Run the Script**:
   - Execute the script using Python:

   ```sh
   python your_script_name.py
   ```

This should resolve the permission issue and allow the script to run correctly.

Certainly! Adding graceful shutdown and suspend-resume functionality to your script involves several steps. You need to:

1. **Save the state of processing**: This includes the list of processed files and any intermediate results.
2. **Check for existing state on startup**: If the script is restarted, it should check for any existing state and resume from where it left off.
3. **Handle interrupts gracefully**: Ensure that the script can handle interrupts (like `Ctrl+C`) and save the state before exiting.

Here's how you can modify your script to include these features:

```python
import os
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import signal
import sys
import json

# Define constants
PDF_DIRECTORY = "./pdf_files"  # Directory containing PDF files
OUTPUT_FILE = "pdf_classification_results.csv"  # Output CSV file
BERT_MODEL = "bert-base-uncased"  # Pre-trained BERT model
CLASSIFIER = "facebook/bart-large-mnli"  # Zero-shot classifier
STATE_FILE = "processing_state.json"  # File to save processing state

# Global variables to keep track of state
processed_files = set()

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=CLASSIFIER)

# Labels for classification
labels = ["science", "technology", "law", "medicine", "literature"]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try to extract text directly
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # If no text is found, use OCR to extract text from images
                image = page.to_image()
                text += pytesseract.image_to_string(image) + "\n"
    return text.strip()

def chunk_text(text, chunk_size=512):
    """Split text into chunks of a specified size."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def classify_pdf_text(file_name, text, labels):
    """Classify the extracted text from a PDF."""
    try:
        chunks = chunk_text(text)
        results = []
        for chunk in chunks:
            result = classifier(chunk, candidate_labels=labels, multi_label=True)
            results.append(result)

        # Aggregate results
        aggregated_labels = set()
        aggregated_scores = {}
        for result in results:
            for label, score in zip(result["labels"], result["scores"]):
                aggregated_labels.add(label)
                if label in aggregated_scores:
                    aggregated_scores[label] = max(aggregated_scores[label], score)
                else:
                    aggregated_scores[label] = score

        return {
            "File Name": file_name,
            "Labels": list(aggregated_labels),
            "Scores": [aggregated_scores[label] for label in aggregated_labels]
        }
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

def save_state(processed_files):
    """Save the state of processed files to a JSON file."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=4)

def load_state():
    """Load the state of processed files from a JSON file."""
    global processed_files
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            processed_files = set(json.load(f))

def signal_handler(sig, frame):
    """Handle interrupt signals and save state before exiting."""
    print("\nInterrupt received, saving state and exiting...")
    save_state(processed_files)
    sys.exit(0)

def main():
    global processed_files

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load existing state
    load_state()

    results = []

    # Process each PDF in the directory
    for file_name in os.listdir(PDF_DIRECTORY):
        file_path = os.path.join(PDF_DIRECTORY, file_name)

        if file_name.endswith(".pdf") and os.path.isfile(file_path) and file_name not in processed_files:
            print(f"Processing: {file_name}")

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            if not text:
                print(f"Could not extract text from {file_name}. Skipping.")
                continue

            # Classify the extracted text
            classification_result = classify_pdf_text(file_name, text, labels)
            if classification_result:
                results.append(classification_result)

            # Mark file as processed
            processed_files.add(file_name)

    # Save results to a CSV file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("File Name,Label,Score\n")
        for result in results:
            for label, score in zip(result["Labels"], result["Scores"]):
                f.write(f"{result['File Name']},{label},{score:.4f}\n")

    # Save final state
    save_state(processed_files)

    print(f"Results saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()

# Initialize Dash app for visualization
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the app
app.layout = html.Div([
    html.H1("PDF Document Scores Visualization", style={"textAlign": "center"}),

    # Slider for classification score threshold
    html.Div([
        html.Label("Minimum Classification Score:"),
        dcc.Slider(
            id="min-score-slider",
            min=0,
            max=1,
            step=0.01,
            value=0.5,  # Default threshold
            marks={i / 10: str(i / 10) for i in range(11)}
        )
    ], style={"padding": "20px"}),

    # Checkboxes for label inclusion/exclusion
    html.Div([
        html.Label("Include Labels:"),
        dcc.Checklist(
            id="include-labels",
            options=[{"label": label, "value": label} for label in labels],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    html.Div([
        html.Label("Exclude Labels:"),
        dcc.Checklist(
            id="exclude-labels",
            options=[{"label": label, "value": label} for label in labels],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    # Button to export filtered results
    html.Div([
        dbc.Button("Export Filtered Results", id="export-button", color="primary", className="me-1")
    ], style={"padding": "20px"}),

    # Graph
    dcc.Graph(id="document-chart"),

    # Hidden div to store filtered data for export
    dcc.Store(id="filtered-data")
])

# Callback to update chart based on thresholds and labels
@app.callback(
    [Output("document-chart", "figure"), Output("filtered-data", "data")],
    [Input("min-score-slider", "value"),
     Input("include-labels", "value"),
     Input("exclude-labels", "value")]
)
def update_chart(min_score, include_labels, exclude_labels):
    # Load CSV data
    data = pd.read_csv(OUTPUT_FILE)

    # Filter data based on thresholds and labels
    filtered_data = data[
        (data["Score"] >= min_score)
    ]
    if include_labels:
        filtered_data = filtered_data[filtered_data["Label"].isin(include_labels)]
    if exclude_labels:
        filtered_data = filtered_data[~filtered_data["Label"].isin(exclude_labels)]

    # Create bar chart
    fig = px.bar(
        filtered_data,
        x="File Name",
        y="Score",
        color="Label",
        hover_data=["Score"],
        title="Filtered Document Scores"
    )
    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Classification Score",
        template="plotly_white"
    )

    # Store filtered data for export
    filtered_data_json = filtered_data.to_json(date_format='iso', orient='split')

    return fig, filtered_data_json

# Callback to export filtered results
@app.callback(
    Output("export-button", "n_clicks"),
    [Input("export-button", "n_clicks")],
    [State("filtered-data", "data")]
)
def export_filtered_results(n_clicks, filtered_data_json):
    if n_clicks:
        filtered_data = pd.read_json(filtered_data_json, orient='split')
        filtered_data.to_csv("filtered_pdf_documents.csv", index=False)
        print("Filtered results exported to filtered_pdf_documents.csv")
    return 0

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```

### Key Changes:
1. **State Management**:
   - Added `STATE_FILE` to keep track of processed files.
   - `save_state` and `load_state` functions to save and load the state of processed files.
   - `processed_files` set to keep track of files that have already been processed.

2. **Signal Handling**:
   - Registered signal handlers for `SIGINT` and `SIGTERM` to handle interrupts gracefully.
   - `signal_handler` function to save the state and exit gracefully on interrupt.

3. **Resume Functionality**:
   - At the start of the script, `load_state` is called to load any existing state.
   - During processing, the script checks if a file has already been processed and skips it if so.

### Setup Instructions:
1. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install pandas plotly dash dash-bootstrap-components pdfplumber pytesseract transformers
   ```

2. **Prepare PDF Directory**:
   Ensure your `pdf_files` directory contains the PDF files you want to process.

3. **Run the Script**:
   Save the script as `process_pdfs.py` and run:
   ```bash
   python3 process_pdfs.py
   ```

4. **View in Browser**:
   Open your browser to the URL displayed in the terminal (usually `http://127.0.0.1:8050`).

### What You Can Do:
- **Identify Relevant Documents**:
  - Use the sliders and checkboxes to explore documents with scores above or below specific thresholds.
  - Quickly identify high-scoring or low-scoring documents.

- **Refine Filtering Criteria**:
  - Visually inspect how thresholds and labels impact document selection.
  - Adjust filters based on the distribution of scores.

- **Export Results**:
  - Export the filtered data to a new CSV file for further analysis or reporting.

### Why Use Dash and Plotly?
- **Interactivity**: Users can explore the data dynamically, making it easier to refine criteria and understand results.
- **Ease of Use**: Minimal coding is required to build complex, interactive visualizations.
- **Extensibility**: Dash apps can be scaled to include advanced filters, additional charts, and even integrated machine learning models.

Let me know if you'd like help adding more advanced features! 😊
