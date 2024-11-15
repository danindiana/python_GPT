### Troubleshooting Guide: Resolving Conflicts Between `fitz` and `PyMuPDF`

If you encounter the error `module 'fitz' has no attribute 'Document'` while using PyMuPDF, it is likely caused by a conflicting `fitz` package that is unrelated to PyMuPDF. Follow this guide to resolve the issue.

---

### **Step-by-Step Troubleshooting**

#### **1. Identify the Problem**
Run the following command to check for installed packages related to `fitz`:
```bash
pip list | grep fitz
```
- If you see `fitz` (e.g., `fitz 0.0.1.dev2`) **alongside** `PyMuPDF`, the `fitz` package is conflicting with the `fitz` module provided by PyMuPDF.

---

#### **2. Uninstall the Conflicting `fitz` Package**
To remove the incorrect `fitz` package, run:
```bash
pip uninstall fitz
```
When prompted, confirm the uninstallation.

---

#### **3. Verify and Reinstall `PyMuPDF`**
Ensure that `PyMuPDF` is correctly installed:
```bash
pip install --force-reinstall pymupdf
```

---

#### **4. Confirm the Correct `fitz` Module**
Run the following Python code to verify that the `fitz` module is now correctly linked to PyMuPDF:
```python
import fitz

# Test for the presence of the Document class
print(dir(fitz))
```
You should see `Document` in the output. If `Document` is not listed, something is still misconfigured.

---

#### **5. Test PyMuPDF Functionality**
Run a simple script to verify that PyMuPDF works as expected:
```python
import fitz

# Replace with a valid PDF path
pdf_path = "/path/to/sample.pdf"

try:
    doc = fitz.Document(pdf_path)
    print(f"Number of pages in PDF: {len(doc)}")
    print("Text from first page:")
    print(doc[0].get_text())
except Exception as e:
    print(f"Error: {e}")
```

---

#### **6. Double-Check for Remaining Conflicts**
Re-check for conflicting packages:
```bash
pip list | grep fitz
```
If you still see a conflicting `fitz` package, repeat steps 2 and 3.

---

### **Root Cause of the Issue**
The `fitz` package (`0.0.1.dev2`) is unrelated to PyMuPDF but shares the same module name (`fitz`). This can lead to Python importing the wrong package, causing attribute errors such as `module 'fitz' has no attribute 'Document'`.

### **Preventing the Issue**
To avoid this in the future:
1. Be cautious when installing packages with similar names.
2. Use virtual environments to isolate project dependencies:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```
3. Always verify package names before installation:
   ```bash
   pip install pymupdf  # Correct package for PyMuPDF
   ```

---

### **Common Commands for Debugging**
- Check installed `fitz` packages:
  ```bash
  pip list | grep fitz
  ```
- Uninstall conflicting `fitz`:
  ```bash
  pip uninstall fitz
  ```
- Reinstall `PyMuPDF`:
  ```bash
  pip install --force-reinstall pymupdf
  ```

---

### **If the Issue Persists**
If you've followed these steps and the issue still occurs:
1. Verify the Python environment (system Python vs. virtual environment).
2. Check for global installations of the `fitz` package:
   ```bash
   pip3 list | grep fitz
   ```
3. Seek help with detailed information about your environment:
   - Python version: `python --version`
   - Installed `fitz` and `PyMuPDF` versions: `pip list | grep fitz`
   - Error traceback.

With this guide, you should be able to resolve and prevent the `fitz` and PyMuPDF conflict effectively. 🎯
