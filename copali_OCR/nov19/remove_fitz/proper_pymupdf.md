You're absolutely correct that `fitz` is often mistakenly assumed to refer to PyMuPDF in some setups, but it can lead to confusion when there are overlapping namespaces or inheritance issues.

Here’s how to handle this and ensure clarity:

---

### **Steps to Mark and Resolve Import Issues**

1. **Explicitly Reference PyMuPDF**:
   When using `fitz` from PyMuPDF, ensure it is clear in the code that this is the intended library. You can do this by:

   ```python
   import fitz  # PyMuPDF library for handling PDFs
   ```

2. **Resolve Namespace Issues**:
   If there's ambiguity because another library or custom module uses `fitz`, you can alias the import:
   ```python
   import fitz as pymupdf  # Alias to clarify it's PyMuPDF
   ```

   Update all calls in the file to use `pymupdf`, such as:
   ```python
   doc = pymupdf.open("example.pdf")
   ```

3. **Mark and Comment the Dependency**:
   Add comments or a block of documentation in the code to warn future developers about the potential confusion:
   ```python
   # Important: Ensure 'fitz' refers to PyMuPDF. 
   # Other libraries may use the same name and cause conflicts.
   import fitz  # PyMuPDF library for handling PDFs
   ```

4. **Test the Dependency**:
   Verify that PyMuPDF is correctly installed and used:
   ```bash
   pip install PyMuPDF
   python -c "import fitz; print(fitz.__doc__)"
   ```
   If `fitz` isn't resolving to PyMuPDF, you may need to clean up your Python environment or clarify imports further.

5. **Use Virtual Environments**:
   If you're encountering inheritance problems due to overlapping libraries, isolate your dependencies in a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   pip install PyMuPDF
   ```

6. **Log Issues in Documentation**:
   Add a README or inline comments to flag this issue for others:
   ```
   # Known Issue: The 'fitz' module from PyMuPDF may conflict with similarly named modules.
   # Ensure you're importing PyMuPDF. Aliasing as 'pymupdf' is recommended for clarity.
   ```

---

### **Next Steps**
- Review and refactor all imports of `fitz` in your project.
- If you identify conflicts from other modules or custom scripts, consider renaming them to avoid collisions.

Let me know if you'd like assistance refactoring the imports or testing the setup!
