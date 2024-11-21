Certainly! To improve performance for `txt` output by incorporating JSON-like optimizations, you can focus on reducing the overhead associated with line-by-line file operations and text manipulations. Here's how to refactor the `main_script.py`:

---

### Key Optimizations for `txt` Output:
1. **Batch Writing**: Write the entire content to the file in a single operation, similar to how JSON data is dumped in one step.
2. **Avoid Line-by-Line Loops**: Minimize iterations for text manipulation by processing the entire content at once.
3. **Buffer Management**: Use larger buffer sizes or disable frequent flushes to reduce I/O overhead.
4. **Efficient String Handling**: Use Python's `str.join()` to combine strings, which is more efficient than repeated concatenation in loops.

---

### Refactored Code for `txt` Output:
Here’s the updated `main_script.py` focusing on performance improvements when the user selects `txt` as the output format:

```python
# Inside the loop where you process each PDF:
if output_format == 'txt':
    # Combine all content into a single string
    combined_text_with_header = f"OCR-like extracted text:\n{combined_text}\n"

    # Use a single write operation to save the file
    with open(output_file, "w") as f:
        f.write(combined_text_with_header)
```

---

### Complete Refactored Example
Here’s the relevant portion of the `main_script.py` with optimizations for `txt` output:

```python
# Save output in the selected format
output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_ocr_output.{output_format}")
if output_format == 'txt':
    # Combine all text into a single string
    combined_text_with_header = f"OCR-like extracted text:\n{combined_text}\n"

    # Write the content in a single operation
    with open(output_file, "w", buffering=8192) as f:  # Use a larger buffer size for fewer flushes
        f.write(combined_text_with_header)
elif output_format == 'csv':
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['Page', 'Content'])
        for i, line in enumerate(combined_text.splitlines(), start=1):
            writer.writerow([i, line])
elif output_format == 'json':
    with open(output_file, "w") as f:
        json.dump({"content": combined_text}, f, indent=4)
```

---

### Why This Works:
1. **Batch Writing**:
   - Combining `combined_text_with_header` into a single string reduces the overhead of multiple `write()` calls, which are expensive in terms of I/O operations.
2. **Efficient Buffering**:
   - By setting a larger buffer size (`buffering=8192`), the script minimizes the frequency of I/O flushes to the disk, reducing latency.
3. **Streamlined String Operations**:
   - Avoiding line-by-line processing reduces the number of string operations, making the script faster and more memory-efficient.

---

### Expected Benefits:
1. **Reduced I/O Overhead**: Fewer write operations mean faster disk interactions.
2. **Better CPU Utilization**: Less time spent on string concatenations and line-by-line iterations.
3. **Improved Scalability**: Handles larger documents more efficiently with reduced memory and CPU usage.

---

This approach makes the `txt` processing pipeline almost as fast as JSON while retaining the simplicity of plain text output.
