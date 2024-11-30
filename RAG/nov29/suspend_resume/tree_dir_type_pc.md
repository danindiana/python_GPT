Certainly! Below is a pseudo code flow diagram in a tree-like structure to represent the refactored script:

```
main()
├── check_directory_permissions(PDF_DIRECTORY)
│   ├── if not exists:
│   │   └── print("Error: Directory '{directory}' does not exist.")
│   │       └── return False
│   ├── if not writable:
│   │   └── print("Error: You do not have write permissions for the directory '{directory}'.")
│   │       └── return False
│   └── return True
├── signal.signal(signal.SIGINT, signal_handler)
├── signal.signal(signal.SIGTERM, signal_handler)
├── load_state()
│   ├── if STATE_FILE exists:
│   │   └── load processed_files from STATE_FILE
├── results = []
├── for file_name in os.listdir(PDF_DIRECTORY):
│   ├── file_path = os.path.join(PDF_DIRECTORY, file_name)
│   ├── if file_name.endswith(".pdf") and os.path.isfile(file_path) and file_name not in processed_files:
│   │   └── print(f"Processing: {file_name}")
│   │   └── text = extract_text_from_pdf(file_path)
│   │       ├── with pdfplumber.open(pdf_path) as pdf:
│   │       │   └── for page in pdf.pages:
│   │       │       ├── page_text = page.extract_text()
│   │       │       ├── if page_text:
│   │       │       │   └── text += page_text + "\n"
│   │       │       └── else:
│   │       │           └── image = page.to_image()
│   │       │               └── text += pytesseract.image_to_string(image) + "\n"
│   │       └── return text.strip()
│   │   └── if not text:
│   │       └── print(f"Could not extract text from {file_name}. Skipping.")
│   │           └── continue
│   │   └── classification_result = classify_pdf_text(file_name, text, labels)
│   │       ├── chunks = chunk_text(text)
│   │       │   ├── words = text.split()
│   │       │   ├── chunks = []
│   │       │   ├── current_chunk = []
│   │       │   ├── for word in words:
│   │       │   │   ├── current_chunk.append(word)
│   │       │   │   ├── if len(" ".join(current_chunk)) >= chunk_size:
│   │       │   │   │   └── chunks.append(" ".join(current_chunk))
│   │       │   │   │       └── current_chunk = []
│   │       │   ├── if current_chunk:
│   │       │   │   └── chunks.append(" ".join(current_chunk))
│   │       │   └── return chunks
│   │       ├── results = []
│   │       ├── for chunk in chunks:
│   │       │   └── result = classifier(chunk, candidate_labels=labels, multi_label=True)
│   │       │       └── results.append(result)
│   │       ├── aggregated_labels = set()
│   │       ├── aggregated_scores = {}
│   │       ├── for result in results:
│   │       │   └── for label, score in zip(result["labels"], result["scores"]):
│   │       │       ├── aggregated_labels.add(label)
│   │       │       ├── if label in aggregated_scores:
│   │       │       │   └── aggregated_scores[label] = max(aggregated_scores[label], score)
│   │       │       └── else:
│   │       │           └── aggregated_scores[label] = score
│   │       └── return {
│   │            "File Name": file_name,
│   │            "Labels": list(aggregated_labels),
│   │            "Scores": [aggregated_scores[label] for label in aggregated_labels]
│   │        }
│   │   └── if classification_result:
│   │       └── results.append(classification_result)
│   │   └── processed_files.add(file_name)
│   │   └── save_state(processed_files)
│   └── else:
│       └── print(f"Skipping non-PDF file or already processed file: {file_name}")
├── if results:
│   └── with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
│       └── f.write("File Name,Label,Score\n")
│       └── for result in results:
│           └── for label, score in zip(result["Labels"], result["Scores"]):
│               └── f.write(f"{result['File Name']},{label},{score:.4f}\n")
│   └── print(f"Results saved to {OUTPUT_FILE}.")
└── else:
    └── print("No new results to save.")
└── save_state(processed_files)

signal_handler(sig, frame)
├── print("\nInterrupt received, saving state and exiting...")
├── save_state(processed_files)
└── sys.exit(0)

save_state(processed_files)
├── with open(STATE_FILE, "w", encoding="utf-8") as f:
│   └── json.dump(list(processed_files), f, indent=4)

load_state()
├── if os.path.exists(STATE_FILE):
│   └── with open(STATE_FILE, "r", encoding="utf-8") as f:
│       └── processed_files = set(json.load(f))

Dash App
├── app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
├── app.layout = html.Div([
│   ├── html.H1("PDF Document Scores Visualization", style={"textAlign": "center"}),
│   ├── html.Div([
│   │   ├── html.Label("Minimum Classification Score:"),
│   │   ├── dcc.Slider(
│   │   │   ├── id="min-score-slider",
│   │   │   ├── min=0,
│   │   │   ├── max=1,
│   │   │   ├── step=0.01,
│   │   │   ├── value=0.5,
│   │   │   ├── marks={i / 10: str(i / 10) for i in range(11)}
│   │   │)
│   │])
│   ├── html.Div([
│   │   ├── html.Label("Include Labels:"),
│   │   ├── dcc.Checklist(
│   │   │   ├── id="include-labels",
│   │   │   ├── options=[{"label": label, "value": label} for label in labels],
│   │   │   ├── value=[],
│   │   │   ├── inline=True
│   │   │)
│   │])
│   ├── html.Div([
│   │   ├── html.Label("Exclude Labels:"),
│   │   ├── dcc.Checklist(
│   │   │   ├── id="exclude-labels",
│   │   │   ├── options=[{"label": label, "value": label} for label in labels],
│   │   │   ├── value=[],
│   │   │   ├── inline=True
│   │   │)
│   │])
│   ├── html.Div([
│   │   ├── dbc.Button("Export Filtered Results", id="export-button", color="primary", className="me-1")
│   │])
│   ├── dcc.Graph(id="document-chart"),
│   ├── dcc.Store(id="filtered-data")
│])
├── @app.callback(
│   ├── [Output("document-chart", "figure"), Output("filtered-data", "data")],
│   ├── [Input("min-score-slider", "value"),
│   ├── Input("include-labels", "value"),
│   ├── Input("exclude-labels", "value")]
│)
│├── def update_chart(min_score, include_labels, exclude_labels):
│   ├── data = pd.read_csv(OUTPUT_FILE)
│   ├── filtered_data = data[(data["Score"] >= min_score)]
│   ├── if include_labels:
│   │   └── filtered_data = filtered_data[filtered_data["Label"].isin(include_labels)]
│   ├── if exclude_labels:
│   │   └── filtered_data = filtered_data[~filtered_data["Label"].isin(exclude_labels)]
│   ├── fig = px.bar(
│   │   ├── filtered_data,
│   │   ├── x="File Name",
│   │   ├── y="Score",
│   │   ├── color="Label",
│   │   ├── hover_data=["Score"],
│   │   ├── title="Filtered Document Scores"
│   │)
│   ├── fig.update_layout(
│   │   ├── xaxis_title="Document Name",
│   │   ├── yaxis_title="Classification Score",
│   │   ├── template="plotly_white"
│   │)
│   ├── filtered_data_json = filtered_data.to_json(date_format='iso', orient='split')
│   └── return fig, filtered_data_json
├── @app.callback(
│   ├── Output("export-button", "n_clicks"),
│   ├── [Input("export-button", "n_clicks")],
│   ├── [State("filtered-data", "data")]
│)
│├── def export_filtered_results(n_clicks, filtered_data_json):
│   ├── if n_clicks:
│   │   └── filtered_data = pd.read_json(filtered_data_json, orient='split')
│   │       └── filtered_data.to_csv("filtered_pdf_documents.csv", index=False)
│   │           └── print("Filtered results exported to filtered_pdf_documents.csv")
│   └── return 0
└── if __name__ == "__main__":
    └── app.run_server(debug=True)
```

This tree-like pseudo code flow diagram provides a clear and structured representation of the script's logic, including directory permission handling, graceful shutdown/suspend-resume functionality, and the Dash app for visualization.
