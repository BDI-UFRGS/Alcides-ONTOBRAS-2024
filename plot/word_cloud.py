import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set the path to the directory containing folders with .csv files
root_dir = 'datasets/BFO'

# Dictionary to store dataset names and their respective lengths
dataset_lengths = {}

# Iterate through all folders and files in the root directory
for folder, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder, file)
            # Read the CSV file
            try:
                df = pd.read_csv(file_path, delimiter=';')
                # Use the file name without extension as the word
                dataset_name = os.path.splitext(file)[0].split('_')[1].upper()
                # Store the length of the dataset
                dataset_lengths[dataset_name] = len(df)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(dataset_lengths)

print(len(dataset_lengths))
# Plot the word cloud
plt.figure(figsize=(10, 5), dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.title('Word Cloud of Dataset Sizes')
plt.tight_layout()
plt.show()