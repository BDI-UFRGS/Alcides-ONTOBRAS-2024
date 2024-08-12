import pandas as pd
import h5py
import numpy as np
from classifier.knn import KNNClassifier
from sklearn.metrics import classification_report
import json
import os
from experiment.experiment import Experiment

class CrossDomainExperiment(Experiment):
    def __init__(self, tlp: str) -> None:
        self.tlp = tlp

        domains_dfs = self._get_domains()

        print(domains_dfs)

        for train_df, test_df in self._build_train_test(domains_dfs):

            x_train, y_train = self._split_X_y(train_df)
            x_test, y_test = self._split_X_y(test_df)

            self.run(x_train, y_train, x_test, y_test)


    def run(self, x_train, y_train, x_test, y_test):
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(x_train['embedding'], y_train)
        predictions = knn.predict(x_test['embedding'])
        
        name = str(x_test['domain'][0]).upper()
        
        print(f'Performing experiment in {self.tlp.upper()}-{name} domain')
        print(np.unique(x_train['dataset']))
        print(np.unique(x_test['dataset']))

        self.check_right_classified(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, y_pred=predictions, name=name)
        self.check_wrong_classified(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, y_pred=predictions, name=name)

        # print(name)
        # self.confusion_matrix(y_true=y_test, y_pred=predictions, name=name)        

        # report = classification_report(y_true=list(y_test), y_pred=list(predictions), target_names=self.labels, output_dict=True, zero_division=0)
        # with open(f'log\\report_cross_domain_{self.tlp}_{name}.json', 'w') as fp:
        #     json.dump(report, fp, indent=4)

    
    def _build_train_test(self, domains: list):
        for i in range(len(domains)):
            train_df = pd.concat(domains[:i] + domains[i+1:])
            test_df = domains[i]
            
            yield train_df, test_df


    def _get_domains(self):

        domains_dict = dict()

        folder_dir = os.path.join('input_dataset', self.tlp)

        for subdir, _, files in os.walk(folder_dir):
            if folder_dir != subdir:
                domain = subdir.split('\\')[2]

                if domain not in domains_dict:
                    domains_dict[domain] = pd.DataFrame()

                for file in files:
                    file_dir = os.path.join(subdir, file)
                    filename = os.path.splitext(os.path.basename(file))[0]
                    filename = filename.split('_')[-1]
                    
                    dataset_df = pd.read_csv(file_dir)

                    h5_file_path = f'embedding_dataset\\{self.tlp}\\{domain}\\{filename}.h5'

                    with h5py.File(h5_file_path, 'r') as h5_file:
                        embeddings = h5_file['embedding'][:]

                    embeddings_list = [embedding for embedding in embeddings]

                    dataset_df['embedding'] = embeddings_list
                    dataset_df['domain'] = [domain.upper()] * len(dataset_df)
                    dataset_df['dataset'] = [filename.upper()] * len(dataset_df)

                    domains_dict[domain] = pd.concat([domains_dict[domain], dataset_df], ignore_index=True)

        domains = []

        for _, value in domains_dict.items():
            domains.append(value)

        return domains



    def _get_df(self, domains: list) -> pd.DataFrame:
        df = pd.DataFrame()

        for domain in domains:

            folder_dir = f'input_dataset\\{self.tlp}\\{domain}'
            for subdir, _, files in os.walk(folder_dir):

                for file in files:
                    file_dir = os.path.join(subdir, file)
                    filename = os.path.splitext(os.path.basename(file))[0]
                    filename = filename.split('_')[-1]
                    dataset_df = pd.read_csv(file_dir)
                    h5_file_path = f'embedding_dataset\\{self.tlp}\\{domain}\\{filename}.h5'

                    with h5py.File(h5_file_path, 'r') as h5_file:
                        embeddings = h5_file['embedding'][:]

                    embeddings_list = [embedding for embedding in embeddings]

                    dataset_df['embedding'] = embeddings_list
                    dataset_df['domain'] = [domain.upper()] * len(dataset_df)
                    dataset_df['dataset'] = [filename.upper()] * len(dataset_df)

                    df = pd.concat([df, dataset_df], ignore_index=True)

        return df
    


