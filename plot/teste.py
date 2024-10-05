# import numpy as np
# import matplotlib.pyplot as plt
# # Repeating the calculation and plotting steps

# # Data provided for three datasets
# data_1 = {
#     "ICEO": 54, "GSSO": 74, "GO": 312, "VO": 302, "ENVO": 305, "OBI": 226, "PROCO": 51,
#     "FOODON": 147, "RBO": 818, "CLO": 57, "UBERON": 26, "AGRO": 30, "MCO": 9, "FBBT": 29,
#     "CL": 22, "PO": 8, "PDRO": 1, "OBIB": 4, "OHMI": 1, "IDO": 3, "OHD": 3, "EUPATH": 1,
#     "ECOCORE": 1, "DOID": 46, "OAE": 1, "CDNO": 2, "INO": 2
# }

# data_2 = {
#     "CHEBI": 156323, "GSSO": 372, "VO": 416, "ICEO": 48, "GO": 388, "OBI": 254, "ENVO": 296,
#     "RBO": 820, "FOODON": 141, "CLO": 53, "AGRO": 31, "UBERON": 28, "OHMI": 2, "PROCO": 55,
#     "MCO": 9, "FBBT": 28, "CL": 22, "PDRO": 1, "PO": 7, "CDNO": 5, "EUPATH": 2, "OBIB": 4,
#     "IDO": 3, "OHD": 3, "ONS": 1, "ECOCORE": 1, "DOID": 45, "OAE": 1, "INO": 2
# }

# data_3 = {
#     "CHEBI": 188665, "ICEO": 45, "FOODON": 129, "GO": 364, "OBI": 233, "GSSO": 172, "VO": 370,
#     "ENVO": 260, "UBERON": 27, "RBO": 682, "AGRO": 33, "OHMI": 2, "PROCO": 47, "MCO": 8,
#     "FBBT": 28, "CL": 23, "CLO": 50, "PDRO": 1, "PO": 5, "CDNO": 5, "OBIB": 4, "IDO": 3,
#     "OHD": 3, "ONS": 1, "EUPATH": 1, "ECOCORE": 1, "DOID": 43, "INO": 2
# }

# # Function to calculate proportions for each dataset
# def calculate_proportion(data, keys):
#     total = sum(data.values())
#     # Ensure we assign 0 to missing keys
#     proportions = {key: data.get(key, 0) / total for key in keys}
#     return proportions

# # Get all unique keys (ontologies) across the datasets
# all_keys = sorted(set(data_1.keys()) | set(data_2.keys()) | set(data_3.keys()))

# # Calculate proportions for the 3 datasets, considering all possible keys
# proportions_1 = calculate_proportion(data_1, all_keys)
# proportions_2 = calculate_proportion(data_2, all_keys)
# proportions_3 = calculate_proportion(data_3, all_keys)

# # Prepare data for plotting
# proportions_common_1 = [proportions_1[key] for key in all_keys]
# proportions_common_2 = [proportions_2[key] for key in all_keys]
# proportions_common_3 = [proportions_3[key] for key in all_keys]

# # Create a plot
# x = [1, 2, 3]  # Datasets

# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot lines for each ontology
# for i, ontology in enumerate(all_keys):
#     ax.plot(x, [proportions_common_1[i], proportions_common_2[i], proportions_common_3[i]], label=ontology)

# # Labeling
# ax.set_xlabel('Datasets')
# ax.set_ylabel('Proportion')
# ax.set_title('Ontology Proportions Across Datasets')
# ax.set_xticks(x)
# ax.set_xticklabels(['Dataset 1', 'Dataset 2', 'Dataset 3'])

# # Show legend and plot
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

import os
import json
import matplotlib.pyplot as plt

# Path to the folder containing JSON files
folder_path = 'log\wrong\chebi'

# Function to read all JSON files from a folder
def load_json_files(folder_path):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                data_list.append(data)
    return data_list

# Function to calculate proportions for each dataset
def calculate_proportion(data, keys):
    total = sum(data.values())
    # Ensure we assign 0 to missing keys
    proportions = {key: data.get(key, 0) / total for key in keys}
    return proportions

# Load all datasets from the folder
datasets = load_json_files(folder_path)

# Get all unique keys (ontologies) across all datasets
all_keys = sorted(set().union(*[dataset.keys() for dataset in datasets]))

# Calculate proportions for each dataset
proportions = [calculate_proportion(dataset, all_keys) for dataset in datasets]

# Prepare data for plotting
proportions_common = [[proportion[key] for key in all_keys] for proportion in proportions]

# Create a plot
x = range(1, len(datasets) + 1)  # Datasets indices (1, 2, 3, ..., n)

fig, ax = plt.subplots(figsize=(12, 8))

# Plot lines for each ontology
for i, ontology in enumerate(all_keys):
    ax.plot(x, [proportions_common[dataset_idx][i] for dataset_idx in range(len(datasets))], label=ontology)

# Labeling
ax.set_xlabel('Datasets')
ax.set_ylabel('Proportion')
ax.set_title('Ontology Proportions Across Datasets')
# ax.set_xticks(x)
# ax.set_xticklabels([f'Dataset {i+1}' for i in range(len(datasets))])

# Show legend and plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()