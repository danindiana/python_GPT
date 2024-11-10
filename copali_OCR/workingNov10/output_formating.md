### Similarity Scoring Thresholds

To determine appropriate thresholds for similarity scores to decide when to consider images visually similar, you need to consider the context and the specific use case. Here are some general guidelines:

1. **High Similarity (90-100%)**:
   - **Threshold**: 0.90 to 1.00
   - **Interpretation**: The images are almost identical. This could be useful for detecting duplicates or very similar images.

2. **Moderate Similarity (70-90%)**:
   - **Threshold**: 0.70 to 0.90
   - **Interpretation**: The images are similar but not identical. This could be useful for grouping similar images or identifying variations of the same theme.

3. **Low Similarity (50-70%)**:
   - **Threshold**: 0.50 to 0.70
   - **Interpretation**: The images are somewhat similar but have significant differences. This could be useful for identifying related but distinct images.

4. **Very Low Similarity (0-50%)**:
   - **Threshold**: 0.00 to 0.50
   - **Interpretation**: The images are not similar. This could be useful for filtering out unrelated images.

### Output Formatting

To provide options for different output formats (e.g., JSON, CSV) or structuring the output in a more meaningful way for analysis, you can implement a flexible output system. Here’s how you can do it:

#### JSON Output

```python
import json

def save_output_as_json(output_dir, pdf_file, ocr_text, similarity_scores):
    output_file = os.path.join(output_dir, f"{pdf_file}_output.json")
    output_data = {
        "pdf_file": pdf_file,
        "ocr_text": ocr_text,
        "similarity_scores": similarity_scores
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"JSON output saved to {output_file}")
```

#### CSV Output

```python
import csv

def save_output_as_csv(output_dir, pdf_file, ocr_text, similarity_scores):
    output_file = os.path.join(output_dir, f"{pdf_file}_output.csv")
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_file", "ocr_text", "similarity_scores"])
        writer.writerow([pdf_file, ocr_text, similarity_scores])
    print(f"CSV output saved to {output_file}")
```

#### Integration into Main Script

You can integrate these functions into your main script by adding a user prompt to choose the output format:

```python
# Ask the user for the desired output format
output_format = input("Enter the desired output format (json/csv): ").lower()

# Process each PDF file in the input directory
for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)
    images, pdf_text, ocr_text = extract_images_and_text_ocr(pdf_path, resize_factor=2)

    print(f"Processing images for {pdf_file}...")

    # Save OCR-like text to a file in the output directory
    output_file = os.path.join(output_dir, f"{pdf_file}_ocr_output.txt")
    with open(output_file, "w") as f:
        f.write("OCR-like extracted text:\n")
        f.write(ocr_text)

    print(f"\nOCR-like extracted text saved to {output_file}")

    # Process images with a batch size of 1 to prevent out-of-memory errors
    all_image_embeddings = []
    if images:
        for i in range(0, len(images), 1):  # Batch size reduced to 1
            image_batch = images[i:i + 1]
            batch_images = processor.process_images(image_batch).to(device)
            
            with torch.no_grad():
                image_embeddings = model(**batch_images)
                all_image_embeddings.append(image_embeddings)
            
            torch.cuda.empty_cache()

        if all_image_embeddings:
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        else:
            all_image_embeddings = None
            print("No image embeddings were created.")
    else:
        all_image_embeddings = None
        print("No images found in the PDF for processing.")

    # Use OCR text if direct text extraction was empty
    if not pdf_text.strip() and ocr_text.strip():
        pdf_text = ocr_text

    # Check if there is text content to process
    if pdf_text.strip():
        print("Processing text...")
        # Dynamically split text into manageable chunks based on max_chunk_size
        text_chunks = split_text_into_chunks(pdf_text, max_chunk_size)
        similarity_scores = []
        skip_due_to_length = False

        for chunk in text_chunks:
            if len(chunk.split()) > max_sequence_length:
                print(f"Skipping file {pdf_file} due to chunk length exceeding {max_sequence_length}")
                skip_due_to_length = True
                skipped_files.append(pdf_file)
                break

            try:
                # Proceed with model processing for valid chunks
                queries = [chunk]
                batch_queries = processor.process_queries(queries).to(device)

                with torch.no_grad():
                    query_embeddings = model(**batch_queries)
                    torch.cuda.empty_cache()

                    if all_image_embeddings is not None:
                        scores = processor.score_multi_vector(query_embeddings, all_image_embeddings)
                        similarity_scores.append(scores[0].mean().item())
            except torch.cuda.OutOfMemoryError:
                print("Skipping due to CUDA memory issue.")
                torch.cuda.empty_cache()
                skip_due_to_length = True
                skipped_files.append(pdf_file)
                break

        if skip_due_to_length:
            continue

        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            print(f"Average Similarity Score for {pdf_file}: {avg_score:.4f}")
        else:
            print("No similarity scores were calculated.")
    else:
        print("No text found in the PDF for processing.")

    # Save output based on the chosen format
    if output_format == "json":
        save_output_as_json(output_dir, pdf_file, ocr_text, similarity_scores)
    elif output_format == "csv":
        save_output_as_csv(output_dir, pdf_file, ocr_text, similarity_scores)
    else:
        print("Invalid output format specified. Skipping output saving.")

# Final memory cleanup
torch.cuda.empty_cache()

# Display the list of skipped files
if skipped_files:
    print("\nThe following files were skipped due to exceeding the maximum sequence length:")
    for skipped_file in skipped_files:
        print(skipped_file)
else:
    print("\nNo files were skipped.")
```

### Summary

- **Similarity Scoring Thresholds**: Defined thresholds for different levels of similarity.
- **Output Formatting**: Added support for JSON and CSV output formats based on user input.
- **Integration**: Integrated the output formatting into the main script.

This approach ensures that the output is both flexible and meaningful for further analysis.
