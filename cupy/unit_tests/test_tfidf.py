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
