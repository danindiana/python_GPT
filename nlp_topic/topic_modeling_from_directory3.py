import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

# Source and target directories
source_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/apache_org/"
target_directory = "/media/walter/7514e32b-65c9-4a64-a233-5db2311455f4/topic_extracts/"

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Read and preprocess text data from files in the source directory
corpus = []
file_names = []

total_files = len(os.listdir(source_directory))
processed_files = 0
skipped_files = 0

for filename in os.listdir(source_directory):
    print(f"Processing file {processed_files + 1}/{total_files}: {filename}")
    try:
        with open(os.path.join(source_directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
        corpus.append(text)
        file_names.append(filename)
        processed_files += 1
    except (UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Skipping file: {filename}, Error: {str(e)}")
        skipped_files += 1
        continue

# Check if there are any files to process
if processed_files == 0:
    print("No valid files to process. Exiting.")
    exit()

# Create a TF-IDF weighted Document-Term Matrix
try:
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_dtm = tfidf_vectorizer.fit_transform(corpus)
except Exception as e:
    print(f"Error during TF-IDF processing: {e}")
    exit()

# Apply NMF topic modeling
try:
    num_topics = 100
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_dtm = nmf_model.fit_transform(tfidf_dtm)
except Exception as e:
    print(f"Error during NMF processing: {e}")
    exit()

# Print top words for each topic and save results to the target directory
feature_names = tfidf_vectorizer.get_feature_names_out()
top_words = []

for topic_idx, topic in enumerate(nmf_model.components_):
    top_words.append([feature_names[i] for i in topic.argsort()[:-100 - 1:-1]])

topics_df = pd.DataFrame({'File': file_names, 'Topic': nmf_dtm.argmax(axis=1), 'Top Words': top_words})

# Function to sanitize file names
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

# Save the results to the target directory
for index, row in topics_df.iterrows():
    filename = sanitize_filename(row['File'])
    topic = row['Topic']
    top_words_str = ', '.join(row['Top Words'])

    result_text = f"File: {filename}\nTopic: {topic}\nTop Words: {top_words_str}\n\n"
    result_file_path = os.path.join(target_directory, f"result_{filename}")

    try:
        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            result_file.write(result_text)
        print(f"Saved result to: {result_file_path}")
    except Exception as e:
        print(f"Error saving file {filename}: {str(e)}")

if processed_files > 0:
    print("Results saved to the target directory.")
else:
    print("No results were saved as no files were successfully processed.")
