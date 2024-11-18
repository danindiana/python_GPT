The error suggests that Tesseract cannot find the `eng.traineddata` file, even though the `TESSDATA_PREFIX` environment variable is set. This typically happens when:

1. The `TESSDATA_PREFIX` is pointing to the wrong directory.
2. The `eng.traineddata` file is missing or not properly installed.
3. There’s a permissions issue accessing the directory.

Let’s fix it step by step.

---

### **Step 1: Verify TESSDATA_PREFIX**
Ensure the `TESSDATA_PREFIX` variable points to the directory containing the `eng.traineddata` file. Run the following:

```bash
echo $TESSDATA_PREFIX
ls /usr/share/tesseract-ocr/5/tessdata
```

Expected output:
- The `echo` command should display `/usr/share/tesseract-ocr/5/`.
- The `ls` command should list files like `eng.traineddata`.

If `eng.traineddata` is not listed, install it:
```bash
sudo apt install tesseract-ocr-eng
```

---

### **Step 2: Update the Environment Variable**
Set the correct `TESSDATA_PREFIX` in the terminal to ensure it works consistently across sessions:
```bash
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/
```

To make this permanent, add it to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):
```bash
echo 'export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/' >> ~/.bashrc
source ~/.bashrc
```

---

### **Step 3: Test Tesseract**
Test if Tesseract works from the terminal:
```bash
tesseract --tessdata-dir /usr/share/tesseract-ocr/5/tessdata test_image.png output_file -l eng
```
- Replace `test_image.png` with a valid image file containing text.
- Replace `output_file` with the desired text output filename.

If this works, Tesseract is properly configured. If it still fails, check for missing or broken dependencies:
```bash
sudo apt update
sudo apt --fix-broken install
sudo apt install tesseract-ocr
```

---

### **Step 4: Adjust Your Code**
Ensure `TESSDATA_PREFIX` is correctly set in your script:
```python
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'
```

Confirm the file exists in Python:
```python
tessdata_path = os.path.join(os.environ['TESSDATA_PREFIX'], 'tessdata')
if not os.path.exists(os.path.join(tessdata_path, 'eng.traineddata')):
    raise FileNotFoundError(f"The file eng.traineddata is missing in {tessdata_path}. Please install the Tesseract language data.")
```

---

### **Step 5: Debug Permissions**
If the file exists but Tesseract cannot access it, check permissions:
```bash
ls -l /usr/share/tesseract-ocr/5/tessdata/eng.traineddata
```

Ensure the user running the script has read access:
```bash
sudo chmod 644 /usr/share/tesseract-ocr/5/tessdata/eng.traineddata
```

---

### **Step 6: Re-run the Script**
After completing the steps above, re-run the script. If the issue persists, let me know the output of the following commands:
1. `echo $TESSDATA_PREFIX`
2. `ls /usr/share/tesseract-ocr/5/tessdata`

The issue lies in the `TESSDATA_PREFIX` environment variable. It points to `/usr/share/tesseract-ocr/4.00/tessdata/`, but the actual directory containing `eng.traineddata` and other trained data is `/usr/share/tesseract-ocr/5/tessdata/`.

Here’s how to resolve the problem:

---

### **Steps to Fix**

1. **Update the `TESSDATA_PREFIX` Variable**:
   Temporarily set the correct path in your current shell session:
   ```bash
   export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/
   ```

   To make it permanent, add this to your shell configuration file (`~/.bashrc` or `~/.zshrc`):
   ```bash
   echo 'export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Update the Script**:
   In your Python code, update the `TESSDATA_PREFIX` setting:
   ```python
   os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'
   ```

3. **Test Tesseract Again**:
   Verify Tesseract can now find the `eng.traineddata` file by running:
   ```bash
   tesseract --tessdata-dir /usr/share/tesseract-ocr/5/tessdata test_image.png output_file -l eng
   ```
   Replace `test_image.png` with a valid image file containing text. If this works, Tesseract is properly configured.

4. **Re-run Your Script**:
   Execute your Python script again. The OCR process should now correctly use the `eng.traineddata` file.

---

### **Additional Debugging if the Issue Persists**
If you still encounter the same error:
1. **Check Permissions**:
   Ensure the Tesseract directory and its files are readable:
   ```bash
   sudo chmod -R 755 /usr/share/tesseract-ocr/5/tessdata
   ```

2. **Reinstall `eng.traineddata`**:
   If `eng.traineddata` is corrupt or missing, reinstall it:
   ```bash
   sudo apt update
   sudo apt install tesseract-ocr-eng
   ```

3. **Explicit Path in Tesseract Commands**:
   Modify your Python script to pass the `--tessdata-dir` flag to Tesseract explicitly:
   ```python
   pytesseract.pytesseract.tesseract_cmd = 'tesseract'
   custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata"'
   data = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
   ```

After these steps, Tesseract should locate the `eng.traineddata` file and function correctly. 

To explicitly set the `--tessdata-dir` flag, you need to modify the sections of your script where `pytesseract.image_to_data` or other OCR-related functions are called. Here's how and where to apply this modification:

### **Update the `extract_text_with_boxes` Function**

Locate this function in your script:

```python
def extract_text_with_boxes(image):
    """Extract text with bounding boxes."""
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        return data  # Dictionary containing text and bounding box information
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract OCR error: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}
```

Modify it to include the `--tessdata-dir` configuration:

```python
def extract_text_with_boxes(image):
    """Extract text with bounding boxes."""
    try:
        # Explicitly pass the tessdata directory
        custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata"'
        data = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
        return data  # Dictionary containing text and bounding box information
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract OCR error: {e}")
        return {'text': [], 'left': [], 'top': [], 'width': [], 'height': []}
```

---

### **Update Other Functions Using Tesseract**

If you have any other calls to Tesseract in your script (e.g., `pytesseract.image_to_string`), apply the same change. For example:

```python
page_ocr_text = pytesseract.image_to_string(processed_image)
```

Change to:

```python
custom_config = r'--tessdata-dir "/usr/share/tesseract-ocr/5/tessdata"'
page_ocr_text = pytesseract.image_to_string(processed_image, config=custom_config)
```

---

### **Why This Works**

- The `--tessdata-dir` flag explicitly tells Tesseract where to look for the language files.
- This bypasses potential issues with the `TESSDATA_PREFIX` environment variable.

---

### **Summary of Changes**

1. Locate all Tesseract function calls in your script (`pytesseract.image_to_data`, `pytesseract.image_to_string`, etc.).
2. Add the `--tessdata-dir` configuration using the `config` parameter.
3. Ensure the path matches your trained data location (`/usr/share/tesseract-ocr/5/tessdata`).

After making these updates, your script should correctly find and use the `eng.traineddata` file. Let me know if you encounter further issues!
