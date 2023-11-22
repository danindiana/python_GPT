import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

# Source and target directories
source_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/apache_org/" #"source_directory"
target_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/topic_extracts/"  #"target_directory"

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Step 1: Read and preprocess text data from files in the source directory
corpus = []
file_names = []

for filename in os.listdir(source_directory):
    with open(os.path.join(source_directory, filename), 'r', encoding='utf-8') as file:
        try:
            text = file.read()
        except UnicodeDecodeError:
            # If decoding with utf-8 fails, try latin-1
            with open(os.path.join(source_directory, filename), 'r', encoding='latin-1') as alt_file:
                            text = alt_file.read()

        corpus.append(text)
        file_names.append(filename)

# Step 2: Create a TF-IDF weighted Document-Term Matrix
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)

# Step 3: Apply NMF topic modeling
num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=1)
nmf_dtm = nmf_model.fit_transform(tfidf_dtm)

# Step 4: Print top words for each topic and save results to the target directory
feature_names = tfidf_vectorizer.get_feature_names_out()
top_words = []

for topic_idx, topic in enumerate(nmf_model.components_):
    top_words.append([feature_names[i] for i in topic.argsort()[:-100 - 1:-1]])

topics_df = pd.DataFrame({'File': file_names, 'Topic': nmf_dtm.argmax(axis=1), 'Top Words': top_words})

# Save the results to the target directory
for index, row in topics_df.iterrows():
    filename = row['File']
    topic = row['Topic']
    top_words = ', '.join(row['Top Words'])
    
    result_text = f"File: {filename}\nTopic: {topic}\nTop Words: {top_words}\n\n"
    
    with open(os.path.join(target_directory, f"result_{filename}"), 'w', encoding='utf-8') as result_file:
        result_file.write(result_text)

print("Results saved to the target directory.")
