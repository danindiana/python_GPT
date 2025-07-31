```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffd8d8', 'edgeLabelBackground':'#ffffff'}}}%%
flowchart TD
    subgraph Crawler["PDF Crawler Program"]
        A[Start Crawler] --> B[Load JSON config]
        B --> C[Initialize queues & workers]
        C --> D[Crawl worker: Fetch URLs]
        D --> E{Is PDF?}
        E -->|Yes| F[Add to PDF queue]
        E -->|No| G[Parse HTML for links]
        G --> H[Add new URLs to crawl queue]
        F --> I[PDF worker: Download PDF]
        I --> J[Save to download directory]
    end

    subgraph Processor["PDF Processor Program"]
        K[Start Processor] --> L[Discover PDFs in directory]
        L --> M[Manager: Spawn worker processes]
        M --> N[Worker: Process single PDF]
        N --> O{Valid PDF?}
        O -->|Yes| P[Extract text and references]
        O -->|No| Q[Mark as error]
        P --> R[Find DOIs/URLs in text]
        R --> S[Output JSON results]
    end

    J -->|Saves PDFs to| SharedDirectory[("Shared Directory\n(pdf_downloads/)")]
    SharedDirectory --> L
    S -->|Creates| Results[("Results JSON File\n(references_*.json)")]

    style Crawler fill:#ffd8d8,stroke:#333
    style Processor fill:#d8e8ff,stroke:#333
    style SharedDirectory fill:#f0f0f0,stroke-dasharray:5
    style Results fill:#f0f0f0,stroke-dasharray:5
```

How They Work Together:
Crawler Program (Left):

Starts with a JSON config of URLs to crawl

Uses async workers to crawl web pages and find PDFs

Downloads found PDFs to a shared directory (pdf_downloads/)

Processor Program (Right):

Scans the shared directory for PDFs

Uses a manager-worker pattern to process each PDF

Each worker extracts text, references, DOIs and URLs

Results are aggregated into a JSON file

Data Flow:

The crawler populates the shared directory with PDFs

The processor consumes these PDFs to extract academic references

This creates a pipeline where the crawler gathers content and the processor analyzes it

Key Features Shown:

Parallel processing in both programs (async in crawler, multiprocessing in processor)

Fault tolerance (error handling in both)

Separation of concerns (crawling vs content analysis)

Shared filesystem as integration point

