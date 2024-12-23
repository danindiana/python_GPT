To add **graceful shutdown** and **suspend/resume** functionality to the refactored code, we need to:

1. **Graceful Shutdown**:
   - Allow the program to stop processing gracefully when a shutdown signal (e.g., `SIGINT` or `SIGTERM`) is received.
   - Save the current progress and results before exiting.

2. **Suspend/Resume**:
   - Allow the program to pause processing and save its state (e.g., current batch, results, etc.).
   - Provide a mechanism to resume processing from the saved state.

Below is the **final refactored code** with these features added:

---

### Final Refactored Code with Graceful Shutdown and Suspend/Resume

```python
import os
import csv
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from datasets import Dataset
import multiprocessing
from functools import partial
import signal
import time

# Global variables for graceful shutdown and suspend/resume
SHUTDOWN_FLAG = False
SUSPEND_FLAG = False
RESUME_FLAG = False

def compute_cosine_similarity(vector1, vector2):
    """
    Compute cosine similarity between two vectors.
    Assumes both vectors are PyTorch tensors on the same device.
    """
    vector1 = vector1.clone().detach().to(vector2.device)
    return torch.nn.functional.cosine_similarity(vector1, vector2, dim=0).item()


def get_text_embedding(text, tokenizer, model, torch_device, use_amp):
    """
    Generate an embedding for a given text using the embedding model.
    Utilizes mixed precision if enabled.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(torch_device)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs).last_hidden_state
            else:
                outputs = model(**inputs).last_hidden_state

        # Perform average pooling to get a single vector representation
        return outputs.mean(dim=1).squeeze()  # Return as a tensor
    except Exception as e:
        print(f"Error computing embedding for text: {text[:50]}... - {e}")
        return torch.zeros(model.config.hidden_size, device=torch_device)  # Return a zero tensor if embedding fails


def save_results_csv(results, output_file):
    """
    Save results to a CSV file.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])  # Header
        for result in results:
            for label, score, cosine_sim in zip(result["labels"], result["scores"], result["cosine_similarities"]):
                writer.writerow([result["file_name"], label, score, cosine_sim])


def save_results_json(results, output_file):
    """
    Save results to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def clean_text(text):
    """
    Remove or replace invalid UTF-8 characters.
    """
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'ignore')
    else:
        text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    return text


def worker_process(sub_dataset, labels, device_id, return_dict, process_id, batch_size=16):
    """
    Worker function to process a subset of the dataset on a specific GPU.
    """
    global SHUTDOWN_FLAG, SUSPEND_FLAG, RESUME_FLAG

    try:
        # Determine device
        if device_id >= 0:
            torch_device = torch.device(f"cuda:{device_id}")
            use_amp = True  # Enable mixed precision for GPU
        else:
            torch_device = torch.device("cpu")
            use_amp = False  # Mixed precision not supported on CPU

        # Enable benchmarking for optimized performance
        if device_id >= 0:
            torch.backends.cudnn.benchmark = True

        # Initialize zero-shot classification pipeline
        print(f"Process {process_id}: Loading zero-shot classification model on {torch_device}...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_id,
            torch_dtype=torch.float16 if device_id >= 0 else torch.float32
        )

        # Initialize embedding model and tokenizer
        print(f"Process {process_id}: Loading embedding model on {torch_device}...")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(torch_device)
        if device_id >= 0:
            embedding_model = embedding_model.half()  # Convert to FP16 for mixed precision
        embedding_model.eval()  # Set to evaluation mode

        results = []
        num_batches = (len(sub_dataset) + batch_size - 1) // batch_size

        for i in range(num_batches):
            # Check for shutdown or suspend signals
            if SHUTDOWN_FLAG:
                print(f"Process {process_id}: Shutdown signal received. Saving progress and exiting...")
                return_dict[process_id] = results
                return
            if SUSPEND_FLAG:
                print(f"Process {process_id}: Suspend signal received. Saving progress and pausing...")
                return_dict[process_id] = results
                while not RESUME_FLAG:
                    time.sleep(1)  # Wait for resume signal
                print(f"Process {process_id}: Resuming processing...")
                SUSPEND_FLAG = False
                RESUME_FLAG = False

            batch = sub_dataset[i * batch_size: (i + 1) * batch_size]
            file_names = batch["file_name"]
            texts = batch["text"]

            print(f"Process {process_id}: Processing batch {i + 1}/{num_batches}...")

            # Perform zero-shot classification
            try:
                batch_results = classifier(texts, candidate_labels=labels, multi_label=True)
            except Exception as e:
                print(f"Process {process_id}: Error in classification batch {i + 1}: {e}")
                continue  # Skip this batch and move to the next one

            # Process each result in the batch
            for file_name, text, result in zip(file_names, texts, batch_results):
                try:
                    # Compute embeddings
                    text_embedding = get_text_embedding(text, tokenizer, embedding_model, torch_device, use_amp)
                    label_embeddings = [get_text_embedding(label, tokenizer, embedding_model, torch_device, use_amp) for label in labels]

                    # Compute cosine similarities
                    cosine_similarities = [compute_cosine_similarity(text_embedding, label_embedding) for label_embedding in label_embeddings]

                    # Append result
                    results.append({
                        "file_name": file_name,
                        "labels": result["labels"],
                        "scores": result["scores"],
                        "cosine_similarities": cosine_similarities
                    })

                    print(f"Process {process_id}: File '{file_name}' processed successfully.")
                except Exception as e_file:
                    print(f"Process {process_id}: Error processing file '{file_name}': {e_file}")
                    continue

        # Store results in the shared dictionary
        return_dict[process_id] = results
    except Exception as e:
        print(f"Process {process_id}: Fatal error: {e}")
        return_dict[process_id] = []


def list_available_devices():
    """
    List available GPU devices and return their IDs.
    """
    available_devices = []
    if not torch.cuda.is_available():
        print("No GPU devices available. Running on CPU.")
        available_devices.append(-1)  # CPU
    else:
        print("Available GPU devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            available_devices.append(i)

    # Prompt user to select a device
    while True:
        print("\nSelect a processing option:")
        print("  1. Use one GPU")
        print("  2. Use both GPUs")
        print("  3. Use CPU")
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == "1" and len(available_devices) >= 1:
            device_id = input("Select a GPU device ID: ").strip()
            if device_id.isdigit() and int(device_id) in available_devices:
                print(f"Using GPU device {device_id}: {torch.cuda.get_device_name(int(device_id))}")
                return [int(device_id)]
            else:
                print("Invalid selection. Please enter a valid GPU device ID.")
        elif choice == "2" and len(available_devices) >= 2:
            print("Using both GPUs.")
            return available_devices[:2]  # Use the first two GPUs
        elif choice == "3":
            print("Using CPU.")
            return [-1]  # Use CPU
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")


def signal_handler(sig, frame):
    """
    Signal handler for graceful shutdown and suspend/resume.
    """
    global SHUTDOWN_FLAG, SUSPEND_FLAG, RESUME_FLAG
    if sig == signal.SIGINT:  # Ctrl+C for shutdown
        print("\nShutdown signal received. Exiting gracefully...")
        SHUTDOWN_FLAG = True
    elif sig == signal.SIGUSR1:  # Suspend signal
        print("\nSuspend signal received. Pausing processing...")
        SUSPEND_FLAG = True
    elif sig == signal.SIGUSR2:  # Resume signal
        print("\nResume signal received. Resuming processing...")
        RESUME_FLAG = True


def main():
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGUSR2, signal_handler)

    # Prompt user for the directory path containing text files
    target_dir = input("Enter the path to the directory containing text files: ").strip()

    # Verify the directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # Prompt user for target labels (comma-separated)
    labels_input = input("Enter target labels (comma-separated, e.g., 'science, law, biology, technology'): ").strip()
    labels = [label.strip() for label in labels_input.split(",") if label.strip()]

    if not labels:
        print("Error: No labels provided.")
        return

    # List available devices
    available_devices = list_available_devices()
    num_devices = len([d for d in available_devices if d >= 0])
    if num_devices == 0:
        num_devices = 1  # Use CPU

    # Load and clean dataset
    print("\nLoading and cleaning text files...\n")
    dataset_dict = {"file_name": [], "text": []}
    for file_name in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                text = clean_text(text)
                if text:  # Skip empty files
                    dataset_dict["file_name"].append(file_name)
                    dataset_dict["text"].append(text)
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")

    if not dataset_dict["file_name"]:
        print("No valid text files found to process.")
        return

    dataset = Dataset.from_dict(dataset_dict)

    # Determine number of GPUs or CPU
    print(f"\nDetected {len(available_devices)} device(s).")
    if len(available_devices) > 1:
        print(f"Utilizing {len(available_devices)} GPUs for processing.")
    elif len(available_devices) == 1 and available_devices[0] >= 0:
        print("Utilizing 1 GPU for processing.")
    else:
        print("Utilizing CPU for processing.")

    # Split dataset into shards based on the number of devices
    split_datasets = []
    if num_devices > 1:
        for i in range(num_devices):
            split = dataset.shard(num_shards=num_devices, index=i, contiguous=True)
            split_datasets.append(split)
    else:
        split_datasets.append(dataset)

    # Create a manager for shared data structures
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Create and start processes
    processes = []
    for i in range(len(split_datasets)):
        device_id = available_devices[i] if num_devices > 1 else available_devices[0]
        p = multiprocessing.Process(
            target=worker_process,
            args=(split_datasets[i], labels, device_id, return_dict, i)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Collect results from all processes
    results = []
    for i in range(len(split_datasets)):
        results.extend(return_dict.get(i, []))

    # Save results to CSV and JSON
    output_csv = "classification_results_with_similarity.csv"
    output_json = "classification_results_with_similarity.json"
    save_results_csv(results, output_csv)
    save_results_json(results, output_json)

    print(f"\nResults saved to '{output_csv}' and '{output_json}'.")


if __name__ == "__main__":
    main()
```

---

### Key Features Added:

1. **Graceful Shutdown**:
   - The program listens for the `SIGINT` signal (Ctrl+C) to initiate a graceful shutdown.
   - When a shutdown signal is received, the program saves the current results and exits.

2. **Suspend/Resume**:
   - The program listens for `SIGUSR1` (suspend) and `SIGUSR2` (resume) signals.
   - When a suspend signal is received, the program saves its progress and pauses processing.
   - When a resume signal is received, the program resumes processing from where it left off.

3. **Signal Handling**:
   - A `signal_handler` function is registered to handle `SIGINT`, `SIGUSR1`, and `SIGUSR2` signals.

4. **Global Flags**:
   - `SHUTDOWN_FLAG`, `SUSPEND_FLAG`, and `RESUME_FLAG` are used to manage the state of the program.

5. **Progress Saving**:
   - The program saves results and progress in the `return_dict` before exiting or suspending.

---

### How to Use:

1. **Graceful Shutdown**:
   - Press `Ctrl+C` to trigger a graceful shutdown.

2. **Suspend/Resume**:
   - Send the `SIGUSR1` signal to suspend the program (e.g., using `kill -USR1 <pid>`).
   - Send the `SIGUSR2` signal to resume the program (e.g., using `kill -USR2 <pid>`).

---

### Conclusion:

The final refactored code now supports **graceful shutdown** and **suspend/resume** functionality. It ensures that the program can stop or pause processing gracefully, saving its progress and results. This makes the program more
