The `text_novelty_scorer.py` script is designed to score a collection of text files for novelty based on their content. Here's a step-by-step explanation of how the script works:

1. **Class Definitions**:
   - `TextDocument`: Represents an individual text document. It stores the text, tags (usually the filename), a vector representation of the document, and a novelty score.
   - `NoveltyArchive`: Manages a collection of document vectors and provides a method to calculate the novelty score of a new document based on its distance from the vectors already in the archive.

2. **Doc2Vec Model Training**:
   - `train_doc2vec_model(tagged_data)`: This function takes tagged data (a list of `TaggedDocument` objects) and trains a `Doc2Vec` model. The `Doc2Vec` model is an unsupervised algorithm to generate vector representations for documents, which allows us to compare them semantically.

3. **Reading Text Files**:
   - `read_text_files(directory)`: This function reads all `.txt` files from the given directory, converts the content of each file into a list of words, and wraps it into a `TaggedDocument` object with tags (usually the filename).

4. **User Input**:
   - The script starts by asking the user to input the path to the directory containing the text files to be scored.

5. **Processing Text Files**:
   - The script reads the text files from the provided directory and prepares them for processing.
   - It then trains a `Doc2Vec` model on the prepared data to learn vector representations of the documents.

6. **Scoring for Novelty**:
   - Each document is converted into a vector using the trained `Doc2Vec` model.
   - The `NoveltyArchive` object is used to calculate the novelty score for each document. The novelty score is based on the average cosine distance from the document's vector to its k-nearest neighbors' vectors in the archive.
   - After scoring, each document's vector is added to the archive to be considered as a neighbor for subsequent documents.

7. **Sorting and Output**:
   - The documents are sorted in descending order based on their novelty scores.
   - The script prints out the filename and novelty score for each document.

The script uses the cosine distance to measure novelty, which is a common approach in NLP for comparing the similarity of documents. By considering the nearest neighbors in the vector space, the script assigns higher novelty scores to documents that are less similar to the ones already seen, thus capturing the essence of "novelty" as described in the referenced paper.

The `text_novelty_scorer.py` script is a high-level implementation and may require further refinement or adjustment depending on the specific dataset and the desired granularity of the novelty scoring. It assumes that the text files are preprocessed (e.g., tokenized) if necessary, and that the `Doc2Vec` model parameters are suitable for the dataset at hand.
