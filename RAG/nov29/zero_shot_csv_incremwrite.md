The large `bart` script you are running writes the **CSV file** to the specified output file path (e.g., `classification_results_with_similarity.csv`) **after processing all the files**. 

### **Key Points about Output Writing**

1. **File Writing Logic**:
   - The results are accumulated in a list called `results` as each file is processed:
     ```python
     results.append({
         "file_name": file_name,
         "labels": result["labels"],
         "scores": result["scores"],
         "cosine_similarities": cosine_similarities
     })
     ```
   - Once all files are processed, the `results` list is written to the CSV file in one go:
     ```python
     save_results_csv(results, output_csv)
     ```

2. **Timing of CSV Generation**:
   - The script does **not** write the CSV file incrementally.
   - The CSV file is created only **after all text files** in the directory have been processed.

3. **Output File Location**:
   - The CSV file is saved in the same directory as the script unless a specific path is provided for the `output_csv` variable:
     ```python
     output_csv = "classification_results_with_similarity.csv"
     ```

### **What Happens During Processing?**
- While the files are being processed:
  - The CSV file does not yet exist.
  - Progress and results are printed to the console for each file, but nothing is written to disk until processing is complete.

---

### **Advantages of This Approach**
- Writing the file after all files are processed ensures that:
  - You don't risk partial or corrupted files if the process is interrupted.
  - The output is neatly written in one operation, reducing the chance of file I/O errors.

---

### **What if You Want Incremental Writing?**

If you want the CSV file to be updated incrementally while processing, you can modify the script to write results to the file as they are generated.

1. **Update the `save_results_csv` Function**:
   Modify it to append rows to the CSV file during each iteration:
   ```python
   def save_results_csv_incrementally(result, output_file):
       """Append a single result to the CSV file."""
       with open(output_file, "a", newline="", encoding="utf-8") as f:
           writer = csv.writer(f)
           for label, score, cosine_sim in zip(result["labels"], result["scores"], result["cosine_similarities"]):
               writer.writerow([result["file_name"], label, score, cosine_sim])
   ```

2. **Initialize the CSV File with a Header**:
   At the start of the script, create the CSV file and write the header:
   ```python
   with open(output_csv, "w", newline="", encoding="utf-8") as f:
       writer = csv.writer(f)
       writer.writerow(["File Name", "Label", "Score", "Cosine Similarity"])
   ```

3. **Call the Incremental Function in the Processing Loop**:
   Replace the `results.append(...)` logic with:
   ```python
   save_results_csv_incrementally({
       "file_name": file_name,
       "labels": result["labels"],
       "scores": result["scores"],
       "cosine_similarities": cosine_similarities
   }, output_csv)
   ```

---

### **Pros and Cons of Incremental Writing**

#### **Pros**:
- Partial results are available immediately if the process is interrupted.
- You can monitor progress by inspecting the CSV file as the script runs.

#### **Cons**:
- Slightly slower overall due to frequent file I/O operations.
- File integrity issues could arise if the process is interrupted during a write operation.

---

### **Summary**
The original script writes the CSV file only **after all files are processed**, ensuring a clean and atomic output. If you prefer incremental updates for progress tracking or fault tolerance, you can modify the script as outlined above. Let me know if you'd like help implementing this! 😊
