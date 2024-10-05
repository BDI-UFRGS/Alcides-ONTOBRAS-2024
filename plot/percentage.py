import os
import json
import matplotlib.pyplot as plt

# Directory where the classification report files are stored
folder_path = "log\\percentage\\go"
dataset_name = "GO"

# The classes for which we need to extract the F1-scores
# target_classes = ["independent continuant", "specifically dependent continuant"]  # Replace with your actual class names
target_classes = ["independent continuant", "process"]  # Replace with your actual class names

# Initialize dictionaries to store F1-scores for each class
f1_scores = {class_name: [] for class_name in target_classes}
percentages = []

# Read each file and extract F1-scores
for percentage in range(0, 100):  # From 1 to 99
    file_name = f"{dataset_name}_{percentage}.json"
    file_path = os.path.join(folder_path, file_name)
    # if os.path.isfile(file_path):
    with open(file_path, "r") as f:
        print(file_path)
        # Assuming the file is in JSON format
        report = json.load(f)
        
        # Extract the F1-scores for the target classes
        for class_name in target_classes:
            if class_name in report and 'f1-score' in report[class_name]:
                f1_score = report[class_name]['f1-score']
                f1_scores[class_name].append(f1_score)
            else:
                f1_scores[class_name].append(None)  # Handle missing data
        
        percentages.append(percentage)

# Plotting the F1-scores for each class
plt.figure(figsize=(6, 4))
for class_name in target_classes:
    plt.plot(percentages, f1_scores[class_name], label=f"{class_name.title()}")

# Adding labels and title
plt.xlabel("Percentage of GO data in the training data")
plt.ylabel("F1 Score")
# plt.title("F1 Scores for Target Classes over Different Percentages")
plt.legend(loc='lower right')
plt.title('GO')
plt.ylim((0, 1))
plt.grid(True)
plt.tight_layout()
plt.show()