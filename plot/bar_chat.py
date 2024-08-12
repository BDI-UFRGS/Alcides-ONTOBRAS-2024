import os
import glob
import json
from collections import defaultdict
import matplotlib.pyplot as plt

class Bar:
    def __init__(self) -> None:
        # Specify the directory where the files are located
        # directory = 'log\\0\\'

        # # Use glob to get a list of all files starting with "right_domain_"
        # file_pattern = os.path.join(directory, "wrong_ontology_CHEBI*")
        # files = glob.glob(file_pattern)

        # sum_dict = defaultdict(int)

        # # Iterate through the list of files and read them as JSON
        # for file_path in files:
        #     with open(file_path, 'r') as file:
        #         try:
        #             content = json.load(file)

        #             for key, value in content.items():
        #                 sum_dict[key] += value

        #         except json.JSONDecodeError:
        #             print(f"Error decoding JSON from file: {file_path}")

        rigth_content = None
        
        percent = 80

        ontology = 'CHEBI'

        with open(f'log\\ontology\\{str(percent)}\\right_ontology_{ontology}.json', 'r') as file:
            try:
                rigth_content = json.load(file)

                # for key, value in content.items():
                #     sum_dict[key] += value

            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {'a'}")
        # Convert back to a regular dictionary if needed

        wrong_content = None
        
        with open(f'log\\ontology\\{str(percent)}\\wrong_ontology_{ontology}.json', 'r') as file:
            try:
                wrong_content = json.load(file)

                # for key, value in content.items():
                #     sum_dict[key] += value

            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {'a'}")
        # Convert back to a regular dictionary if needed



        sum_rigth = sum([rigth_content[key] for key in rigth_content])
        sum_wrong = sum([wrong_content[key] for key in wrong_content])
        all_sum = sum_rigth + sum_wrong
        # Print the resulting dictionary
        # print(len(sum_dict))

        # all_sum = sum([sum_dict[key] for key in sum_dict])

        average_dict = {key: rigth_content[key] / all_sum for key in rigth_content}
        sorted_average = dict(sorted(average_dict.items(), key=lambda item: item[1], reverse=False)[-12:])
        # Plotting
        plt.figure(figsize=(5, 3))
        plt.barh([str(key).upper() for key in sorted_average.keys()], list(sorted_average.values()), color='blue')
        # plt.bar(average_dict.keys(), average_dict.values(), color='blue')
        plt.ylabel('Top 12 domain ontologies')
        plt.xlabel('% of examples in the 5-Nearest Neighbors')
        plt.xlim(0, 1)
        plt.title(f'{ontology}', fontsize=12)
        # plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        average_dict = {key: wrong_content[key] / all_sum for key in wrong_content}
        sorted_average = dict(sorted(average_dict.items(), key=lambda item: item[1], reverse=False)[-12:])
        # Plotting
        plt.figure(figsize=(5, 3))
        plt.barh([str(key).upper() for key in sorted_average.keys()], list(sorted_average.values()), color='red')
        # plt.bar(average_dict.keys(), average_dict.values(), color='blue')
        plt.ylabel('Top 12 domain ontologies')
        plt.xlabel('% of examples in the 5-Nearest Neighbors')
        plt.xlim(0, 1)
        plt.title(f'{ontology}', fontsize=12)
        # plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()