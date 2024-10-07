```mermaid
graph TD

    subgraph Program 1: URL Processing and Cosine Similarity Computation
    A[Import Libraries] --> B[Set Up Logging]
    B --> C[Graceful Shutdown Handling]
    C --> D[Select Subdirectory]
    D --> E[Read URLs from File]
    E --> F[Load URLs from Directory]
    F --> G[Load URLs from Single File]
    G --> H[Get Input Choice]
    H --> I[Thread Pool for Async Saving]
    I --> J[Compute Similarity Chunk]
    J --> K[Check Memory]
    K --> L[Compute Parallel Chunks]
    L --> M[Main Program Logic]
    M --> N[Get Input Choice]
    N --> O[Load URLs]
    O --> P[Flatten URLs]
    P --> Q[TF-IDF Vectorization]
    Q --> R[Compute Parallel Chunks]
    R --> S[Log Completion]
    end

    subgraph Program 2: Novelty Score Calculation
    A2[Import Libraries] --> B2[Set Up Logging]
    B2 --> C2[Scan for Files]
    C2 --> D2[Parse Selection]
    D2 --> E2[Select Files]
    E2 --> F2[Lazy Load URLs]
    F2 --> G2[Validate Sparse Matrix]
    G2 --> H2[Lazy Load Sparse Matrices]
    H2 --> I2[Calculate Novelty Scores]
    I2 --> J2[Main Logic]
    J2 --> K2[Scan for .npz Files]
    K2 --> L2[Select .npz Files]
    L2 --> M2[Scan for .txt Files]
    M2 --> N2[Select .txt Files]
    N2 --> O2[Lazy Load URLs]
    O2 --> P2[Lazy Load Sparse Matrices]
    P2 --> Q2[Process Chunks]
    Q2 --> R2[Save Results to CSV]
    R2 --> S2[Log Completion]
    end

    style A fill:#f9d949,stroke:#333,stroke-width:2px
    style A2 fill:#f9d949,stroke:#333,stroke-width:2px
    style S fill:#49f978,stroke:#333,stroke-width:2px
    style S2 fill:#49f978,stroke:#333,stroke-width:2px
```
