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

ontology = 'go'
# Path to the folder containing JSON files
wrong_folder_path = f'log\wrong\{ontology}'
correcly_folder_path = f'log\correctly\{ontology}'

# Function to read all JSON files from a folder
def load_correctly_json_files():
    data_list = []
    for i in range(0, 100):
        file_path = os.path.join(correcly_folder_path, f'right_ontology_{ontology.upper()}_{i}.json')

        with open(file_path, 'r') as file:
                data = json.load(file)
                data_list.append(data)
    return data_list

def load_wrong_json_files():
    data_list = []
    for i in range(0, 100):
        file_path = os.path.join(wrong_folder_path, f'wrong_ontology_{ontology.upper()}_{i}.json')

        with open(file_path, 'r') as file:
                data = json.load(file)
                data_list.append(data)
    return data_list


# Function to calculate proportions for each dataset
def calculate_proportion(data, keys, total):
    # total = sum(data.values())
    # Ensure we assign 0 to missing keys
    proportions = {key: data.get(key, 0) / total for key in keys}
    return proportions

# Load all datasets from the folder
wrong_datasets = load_wrong_json_files()
correcly_datasets = load_correctly_json_files()

target_dataset = correcly_datasets

# Get all unique keys (ontologies) across all datasets
keys = set()

for dataset in wrong_datasets:
    for key in dataset.keys():
        keys.add(key)

for dataset in correcly_datasets:
    for key in dataset.keys():
        keys.add(key)

all_keys = sorted(keys, reverse=False)

total = [sum(wrong_dataset.values()) + sum(correcly_dataset.values()) for wrong_dataset, correcly_dataset in zip(wrong_datasets, correcly_datasets)]

proportions = [calculate_proportion(dataset, all_keys, t) for dataset, t in zip(target_dataset, total)]

print(total[12])

classified_totals = {key: sum(dataset.get(key, 0) for dataset in target_dataset) for key in all_keys}

top_6_ontologies = sorted(classified_totals, key=classified_totals.get, reverse=True)[:10]

# Filter the proportions for only the top 6 ontologies
filtered_keys = [key for key in top_6_ontologies if any(proportions[dataset_idx][key] > 0 for dataset_idx in range(len(wrong_datasets)))]

# Calculate proportions for the filtered ontologies
proportions_common = [[proportion[key] for key in filtered_keys] for proportion in proportions]

# Create a plot
x = range(1, len(wrong_datasets) + 1)  # Datasets indices (1, 2, 3, ..., n)

fig, ax = plt.subplots(figsize=(6, 5))

# Plot lines for each ontology
for i, ontology in enumerate(filtered_keys):
    ax.plot(x, [proportions_common[dataset_idx][i] for dataset_idx in range(len(wrong_datasets))], label=ontology)

# Labeling
ax.set_xlabel('Datasets')
ax.set_ylabel('Proportion')
ax.set_title('Top 6 Ontology Proportions Across Datasets')
ax.set_ylim((0, 1))

# Show legend and plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()







# filtered_keys = [
#     key for key in all_keys
#     if any(proportions[dataset_idx][key] > 0 for dataset_idx in range(len(target_dataset)))
# ]


# # Calculate proportions for each dataset

# # Prepare data for plotting
# proportions_common = [[proportion[key] for key in filtered_keys] for proportion in proportions]

# # Create a plot
# x = range(1, len(target_dataset) + 1)  # Datasets indices (1, 2, 3, ..., n)

# fig, ax = plt.subplots(figsize=(6, 5))

# # Plot lines for each ontology
# # for i, ontology in enumerate(all_keys):
# #     ax.plot(x, [proportions_common[dataset_idx][i] for dataset_idx in range(len(target_dataset))], label=ontology)

# for i, ontology in enumerate(filtered_keys):
#     ax.plot(x, [proportions_common[dataset_idx][i] for dataset_idx in range(len(target_dataset))], label=ontology)



# # Labeling
# ax.set_xlabel('Datasets')
# ax.set_ylabel('Proportion')
# ax.set_title('Ontology Proportions Across Datasets')
# # ax.set_xticks(x)
# # ax.set_xticklabels([f'Dataset {i+1}' for i in range(len(datasets))])
# ax.set_ylim((0, 1))
# # Show legend and plot
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()