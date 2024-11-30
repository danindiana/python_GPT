Yes, Python offers several powerful visualization libraries that can be used to create **interactive visualizations** for sorting and analyzing documents based on scores. A particularly good option for interactive and dynamic exploration is **Plotly** combined with **Pandas**, which can create bar charts or scatter plots that allow users to explore documents above/below selected thresholds.

---

### **Recommended Libraries**
1. **Plotly**:
   - For interactive and visually appealing charts.
   - Ideal for dynamic thresholds, hover functionality, and real-time sorting.
   - Supports interactive filtering.

2. **Matplotlib**:
   - For static bar charts or scatter plots.
   - Simpler but lacks interactive capabilities.

3. **Dash** (a web framework built on Plotly):
   - To create a dashboard where you can adjust sliders or input fields to dynamically update the chart.

---

### **Example: Interactive Visualization with Plotly**

Below is a Python script that reads your CSV file, visualizes the scores for each document, and allows you to dynamically filter documents based on thresholds using a slider.

---

#### **Python Script**

```python
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Load CSV file
CSV_FILE = "classification_results_with_similarity.csv"
data = pd.read_csv(CSV_FILE)

# Initialize Dash app
app = Dash(__name__)

# Layout for the app
app.layout = html.Div([
    html.H1("Document Scores Visualization", style={"textAlign": "center"}),

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

    # Slider for cosine similarity threshold
    html.Div([
        html.Label("Minimum Cosine Similarity:"),
        dcc.Slider(
            id="min-similarity-slider",
            min=0,
            max=1,
            step=0.01,
            value=0.5,  # Default threshold
            marks={i / 10: str(i / 10) for i in range(11)}
        )
    ], style={"padding": "20px"}),

    # Bar chart
    dcc.Graph(id="document-bar-chart")
])

# Callback to update chart based on thresholds
@app.callback(
    Output("document-bar-chart", "figure"),
    [Input("min-score-slider", "value"),
     Input("min-similarity-slider", "value")]
)
def update_chart(min_score, min_similarity):
    # Filter data based on thresholds
    filtered_data = data[
        (data["Score"] >= min_score) & 
        (data["Cosine Similarity"] >= min_similarity)
    ]

    # Create a bar chart
    fig = px.bar(
        filtered_data,
        x="File Name",
        y="Score",
        color="Label",
        hover_data=["Cosine Similarity", "Combined Score"],
        title="Filtered Document Scores"
    )
    fig.update_layout(
        xaxis_title="Document Name",
        yaxis_title="Classification Score",
        template="plotly_white"
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```

---

### **Features of the Script**
1. **Interactive Sliders**:
   - Adjust thresholds for **classification score** and **cosine similarity**.
   - Dynamically filter documents and update the bar chart.

2. **Bar Chart with Hover Data**:
   - Displays the classification score for each document.
   - Color-coded by label for easy identification.
   - Hover over a bar to view additional information, such as cosine similarity and combined score.

3. **Dynamic Updates**:
   - The chart updates in real time as you move the sliders.

---

### **Setup Instructions**
1. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install pandas plotly dash
   ```

2. **Prepare CSV File**:
   Ensure your `classification_results_with_similarity.csv` file contains the following columns:
   - `File Name`
   - `Label`
   - `Score`
   - `Cosine Similarity`
   - `Combined Score` (optional but useful).

3. **Run the Script**:
   Save the script as `visualize_documents.py` and run:
   ```bash
   python3 visualize_documents.py
   ```

4. **View in Browser**:
   Open your browser to the URL displayed in the terminal (usually `http://127.0.0.1:8050`).

---

### **What You Can Do**
- **Identify Relevant Documents**:
  - Use the sliders to explore documents with scores above or below specific thresholds.
  - Quickly identify high-scoring or low-scoring documents.

- **Refine Filtering Criteria**:
  - Visually inspect how thresholds impact document selection.
  - Adjust filters based on the distribution of scores.

---

### **Extending the Script**
- **Add Checkboxes for Inclusion/Exclusion**:
  Allow users to filter by specific labels (e.g., "Include only 'science'").
- **Scatter Plot Option**:
  Replace the bar chart with a scatter plot to compare multiple metrics (e.g., `Score` vs. `Cosine Similarity`).
- **Export Filtered Results**:
  Add functionality to export the filtered data to a new CSV.

---

### **Why Use Dash and Plotly?**
- **Interactivity**: Users can explore the data dynamically, making it easier to refine criteria and understand results.
- **Ease of Use**: Minimal coding is required to build complex, interactive visualizations.
- **Extensibility**: Dash apps can be scaled to include advanced filters, additional charts, and even integrated machine learning models.

Let me know if you'd like help adding more advanced features! 😊

Sure, I can help you extend the script to include additional features such as checkboxes for inclusion/exclusion based on labels, a scatter plot option, and functionality to export filtered results to a new CSV. Below is the enhanced script:

```python
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Load CSV file
CSV_FILE = "classification_results_with_similarity.csv"
data = pd.read_csv(CSV_FILE)

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout for the app
app.layout = html.Div([
    html.H1("Document Scores Visualization", style={"textAlign": "center"}),

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

    # Slider for cosine similarity threshold
    html.Div([
        html.Label("Minimum Cosine Similarity:"),
        dcc.Slider(
            id="min-similarity-slider",
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
            options=[{"label": label, "value": label} for label in data["Label"].unique()],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    html.Div([
        html.Label("Exclude Labels:"),
        dcc.Checklist(
            id="exclude-labels",
            options=[{"label": label, "value": label} for label in data["Label"].unique()],
            value=[],
            inline=True
        )
    ], style={"padding": "20px"}),

    # Dropdown for chart type
    html.Div([
        html.Label("Chart Type:"),
        dcc.Dropdown(
            id="chart-type",
            options=[
                {"label": "Bar Chart", "value": "bar"},
                {"label": "Scatter Plot", "value": "scatter"}
            ],
            value="bar"
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
     Input("min-similarity-slider", "value"),
     Input("include-labels", "value"),
     Input("exclude-labels", "value"),
     Input("chart-type", "value")]
)
def update_chart(min_score, min_similarity, include_labels, exclude_labels, chart_type):
    # Filter data based on thresholds and labels
    filtered_data = data[
        (data["Score"] >= min_score) &
        (data["Cosine Similarity"] >= min_similarity)
    ]
    if include_labels:
        filtered_data = filtered_data[filtered_data["Label"].isin(include_labels)]
    if exclude_labels:
        filtered_data = filtered_data[~filtered_data["Label"].isin(exclude_labels)]

    # Create chart
    if chart_type == "bar":
        fig = px.bar(
            filtered_data,
            x="File Name",
            y="Score",
            color="Label",
            hover_data=["Cosine Similarity", "Combined Score"],
            title="Filtered Document Scores"
        )
        fig.update_layout(
            xaxis_title="Document Name",
            yaxis_title="Classification Score",
            template="plotly_white"
        )
    elif chart_type == "scatter":
        fig = px.scatter(
            filtered_data,
            x="Score",
            y="Cosine Similarity",
            color="Label",
            hover_data=["File Name", "Combined Score"],
            title="Filtered Document Scores"
        )
        fig.update_layout(
            xaxis_title="Classification Score",
            yaxis_title="Cosine Similarity",
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
        filtered_data.to_csv("filtered_documents.csv", index=False)
        print("Filtered results exported to filtered_documents.csv")
    return 0

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
```

### Features of the Enhanced Script

1. **Interactive Sliders**:
   - Adjust thresholds for **classification score** and **cosine similarity**.
   - Dynamically filter documents and update the chart.

2. **Checkboxes for Label Inclusion/Exclusion**:
   - Allow users to filter by specific labels (e.g., "Include only 'science'").

3. **Chart Type Dropdown**:
   - Switch between a bar chart and a scatter plot to compare multiple metrics (e.g., `Score` vs. `Cosine Similarity`).

4. **Export Filtered Results**:
   - Button to export the filtered data to a new CSV file (`filtered_documents.csv`).

### Setup Instructions

1. **Install Dependencies**:
   Install the required libraries:
   ```bash
   pip install pandas plotly dash dash-bootstrap-components
   ```

2. **Prepare CSV File**:
   Ensure your `classification_results_with_similarity.csv` file contains the following columns:
   - `File Name`
   - `Label`
   - `Score`
   - `Cosine Similarity`
   - `Combined Score` (optional but useful).

3. **Run the Script**:
   Save the script as `visualize_documents.py` and run:
   ```bash
   python3 visualize_documents.py
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
