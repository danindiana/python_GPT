import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

# Source and target directories
source_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/tar_text2/" #"source_directory"
target_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/topic_extracts/"  #"target_directory"


# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Step 1: Read and preprocess text data from files in the source directory
corpus = []
file_names = []

# Counters to track progress
total_files = len(os.listdir(source_directory))
processed_files = 0

for filename in os.listdir(source_directory):
    print(f"Processing file {processed_files + 1}/{total_files}: {filename}")
    try:
        with open(os.path.join(source_directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
    except (UnicodeDecodeError, FileNotFoundError):
        # Skip problematic files
        print(f"Skipping file: {filename}")
        continue

    corpus.append(text)
    file_names.append(filename)

    processed_files += 1

# Step 2: Check if there are any files to process
if processed_files == 0:
    print("No valid files to process. Exiting.")
    exit()

# Step 3: Create a TF-IDF weighted Document-Term Matrix
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)

# Step 4: Apply NMF topic modeling
num_topics = 5
nmf_model = NMF(n_components=num_topics, random_state=1)
nmf_dtm = nmf_model.fit_transform(tfidf_dtm)

# Step 5: Print top words for each topic and save results to the target directory
feature_names = tfidf_vectorizer.get_feature_names_out()
top_words = []

for topic_idx, topic in enumerate(nmf_model.components_):
    top_words.append([feature_names[i] for i in topic.argsort()[:-100 - 1:-1]])

topics_df = pd.DataFrame({'File': file_names, 'Topic': nmf_dtm.argmax(axis=1), 'Top Words': top_words})

# Step 6: Save the results to the target directory
for index, row in topics_df.iterrows():
    filename = row['File']
    topic = row['Topic']
    top_words = ', '.join(row['Top Words'])

    result_text = f"File: {filename}\nTopic: {topic}\nTop Words: {top_words}\n\n"
    result_file_path = os.path.join(target_directory, f"result_{filename}")

    print(f"Saving result to: {result_file_path}")  # Print the full file path before saving

    try:
        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            result_file.write(result_text)
        print(f"Saved result to: {result_file_path}")
    except Exception as e:
        print(f"Error saving file {filename}: {str(e)}")

print("Results saved to the target directory.")