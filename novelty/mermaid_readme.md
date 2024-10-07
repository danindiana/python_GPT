Certainly! Below is a Mermaid diagram that explains the flow and order of operations for both programs. The diagram is divided into two sections: one for each program.

### Mermaid Diagram

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

### Explanation

#### Program 1: URL Processing and Cosine Similarity Computation
1. **Import Libraries**: Import necessary libraries.
2. **Set Up Logging**: Configure logging for the program.
3. **Graceful Shutdown Handling**: Set up signal handlers for graceful shutdown.
4. **Select Subdirectory**: Allow the user to select a subdirectory.
5. **Read URLs from File**: Read URLs from a text file.
6. **Load URLs from Directory**: Load URLs from all `.txt` files in a directory.
7. **Load URLs from Single File**: Load URLs from a single file.
8. **Get Input Choice**: Prompt the user to choose between processing a directory or a single file.
9. **Thread Pool for Async Saving**: Set up a thread pool for asynchronous saving.
10. **Compute Similarity Chunk**: Compute cosine similarity for a chunk of the TF-IDF matrix.
11. **Check Memory**: Check available memory and compare it to a user-defined limit.
12. **Compute Parallel Chunks**: Compute cosine similarity in parallel chunks.
13. **Main Program Logic**: Orchestrate the entire process.
14. **Get Input Choice**: Prompt the user again for input choice.
15. **Load URLs**: Load URLs based on the user's choice.
16. **Flatten URLs**: Flatten the URLs into a single list.
17. **TF-IDF Vectorization**: Compute TF-IDF vectors for the URLs.
18. **Compute Parallel Chunks**: Compute cosine similarity in parallel chunks.
19. **Log Completion**: Log the completion of the process.

#### Program 2: Novelty Score Calculation
1. **Import Libraries**: Import necessary libraries.
2. **Set Up Logging**: Configure logging for the program.
3. **Scan for Files**: Scan the directory for files with specific extensions.
4. **Parse Selection**: Parse flexible ranges for file selection.
5. **Select Files**: Allow the user to select files for processing.
6. **Lazy Load URLs**: Lazily load URLs from selected text files.
7. **Validate Sparse Matrix**: Check if a `.npz` file contains a valid sparse matrix.
8. **Lazy Load Sparse Matrices**: Lazily load valid sparse matrices from selected `.npz` files.
9. **Calculate Novelty Scores**: Calculate novelty scores for each chunk of similarity matrices.
10. **Main Logic**: Orchestrate the entire process.
11. **Scan for .npz Files**: Scan the directory for `.npz` files.
12. **Select .npz Files**: Allow the user to select `.npz` files for processing.
13. **Scan for .txt Files**: Scan the directory for `.txt` files.
14. **Select .txt Files**: Allow the user to select `.txt` files for processing.
15. **Lazy Load URLs**: Lazily load URLs from selected `.txt` files.
16. **Lazy Load Sparse Matrices**: Lazily load sparse matrices from selected `.npz` files.
17. **Process Chunks**: Process each chunk of similarity matrix and corresponding URLs to calculate novelty scores.
18. **Save Results to CSV**: Save the novelty scores to a CSV file in chunks.
19. **Log Completion**: Log the completion of the process.

This Mermaid diagram provides a visual representation of the flow and order of operations for both programs, making it easier to understand their structure and functionality.
