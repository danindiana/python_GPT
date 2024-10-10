To create unit tests for the provided code, we can use a testing framework like `unittest` or `pytest`. The goal is to ensure that each function behaves as expected under various conditions. Below are some unit tests that cover different aspects of the code:

### 1. Test `load_text_files` Function
This function loads a chunk of text files from a directory. We can test it by creating a temporary directory with some text files and checking if the function correctly loads the files.

### 2. Test `build_term_doc_matrix` Function
This function creates a sparse term-document matrix using `CountVectorizer`. We can test it by providing a list of documents and checking if the resulting matrix and terms are correct.

### 3. Test `calculate_tfidf_on_gpu` Function
This function calculates TF-IDF on a GPU. We can test it by providing a small term-document matrix and checking if the TF-IDF values are calculated correctly.

### 4. Test `combine_tfidf_results` Function
This function combines the TF-IDF results from two GPUs. We can test it by providing two small TF-IDF matrices and checking if the combined result is correct.

### 5. Test `save_results_to_file` Function
This function saves the TF-IDF results to a file. We can test it by providing a small TF-IDF matrix and checking if the file is correctly written.

### 6. Test `process_text_files` Function
This function processes text files in chunks and calculates TF-IDF. We can test it by providing a directory with text files and checking if the process completes without errors.

### Example Unit Tests Using `unittest`

```python
import unittest
import os
import tempfile
import shutil
import numpy as np
import cupy as cp
import cupyx
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from your_module import load_text_files, build_term_doc_matrix, calculate_tfidf_on_gpu, combine_tfidf_results, save_results_to_file, process_text_files

class TestTFIDF(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = ['file1.txt', 'file2.txt', 'file3.txt']
        self.test_contents = ['hello world', 'hello python', 'python is great']
        for file, content in zip(self.test_files, self.test_contents):
            with open(os.path.join(self.temp_dir, file), 'w') as f:
                f.write(content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_text_files(self):
        documents, file_names = load_text_files(self.temp_dir, 0, 2)
        self.assertEqual(documents, ['hello world', 'hello python'])
        self.assertEqual(file_names, ['file1.txt', 'file2.txt'])

    def test_build_term_doc_matrix(self):
        documents = ['hello world', 'hello python', 'python is great']
        tf_matrix, terms = build_term_doc_matrix(documents)
        self.assertIsInstance(tf_matrix, cupyx.scipy.sparse.csr_matrix)
        self.assertEqual(terms.tolist(), ['great', 'hello', 'is', 'python', 'world'])

    def test_calculate_tfidf_on_gpu(self):
        tf_matrix = csr_matrix([[1, 0, 1], [0, 1, 1]])
        tf_matrix_cupy = cupyx.scipy.sparse.csr_matrix(tf_matrix)
        tf_idf = calculate_tfidf_on_gpu(tf_matrix_cupy, 0)
        self.assertIsInstance(tf_idf, cupyx.scipy.sparse.csr_matrix)

    def test_combine_tfidf_results(self):
        tf_idf_gpu_0 = cupyx.scipy.sparse.csr_matrix([[1, 0], [0, 1]])
        tf_idf_gpu_1 = cupyx.scipy.sparse.csr_matrix([[2, 0], [0, 2]])
        combined_tf_idf = combine_tfidf_results(tf_idf_gpu_0, tf_idf_gpu_1)
        self.assertIsInstance(combined_tf_idf, cupyx.scipy.sparse.csr_matrix)
        self.assertEqual(combined_tf_idf.shape, (4, 2))

    def test_save_results_to_file(self):
        file_names = ['file1.txt', 'file2.txt']
        tf_idf_matrix = cupyx.scipy.sparse.csr_matrix([[1, 0], [0, 1]])
        terms = ['hello', 'world']
        output_file = os.path.join(self.temp_dir, 'test_output.csv')
        save_results_to_file(file_names, tf_idf_matrix, terms, output_file)
        with open(output_file, 'r') as f:
            content = f.read()
        self.assertIn('hello,file1.txt,1.0', content)
        self.assertIn('world,file2.txt,1.0', content)

    def test_process_text_files(self):
        process_text_files(self.temp_dir, chunk_size=2, save_to_disk=False)
        # This test mainly checks if the function runs without errors

if __name__ == '__main__':
    unittest.main()
```

### Notes:
1. **Temporary Directory**: The `setUp` and `tearDown` methods create and clean up a temporary directory with test files.
2. **Assertions**: Each test method includes assertions to check if the function behaves as expected.
3. **GPU Testing**: The tests involving GPU operations (e.g., `calculate_tfidf_on_gpu`) assume that the GPU is available and correctly configured.

### Running the Tests
To run the tests, save the above code in a file (e.g., `test_tfidf.py`) and run it using `python test_tfidf.py`. This will execute all the unit tests and report any failures.
