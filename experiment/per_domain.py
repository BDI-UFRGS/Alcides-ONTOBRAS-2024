import pandas as pd
import h5py
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from classifier.knn import KNNClassifier
from sklearn.metrics import classification_report
import json
import os
from experiment.experiment import Experiment

class PerDomainExperiment(Experiment):
    def __init__(self, tlp: str, domain: str) -> None:
        self.tlp = tlp
        self.domain = domain
        df = self._get_df()
        print(f'Original size: {df.shape}')
        df = df.drop_duplicates(subset=['term', 'definition'], ignore_index=True)
        print(f'After dropduplicate size: {df.shape}')
        X, y = self._split_X_y(df)
        self.run(X, y)
    
    def run(self, X, y):
        for i, x_train, y_train, x_test, y_test in self._stratify(X, y):
            print(f'Performing FOLD {i} experiment in {self.tlp.upper()}-{self.domain.upper()} domain')
            knn = KNNClassifier(n_neighbors=5)
            knn.fit(x_train, y_train)
            predictions = knn.predict(x_test)

            report = classification_report(y_true=y_test, y_pred=predictions, target_names=self.labels, output_dict=True, zero_division=0)
            with open(f'log\\report_{self.tlp}_{self.domain}_FOLD_{i}.json', 'w') as fp:
                json.dump(report, fp, indent=4)

    def _get_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        folder_dir = f'input_dataset\\{self.tlp}\\{self.domain}'
        for subdir, _, files in os.walk(folder_dir):

            for file in files:
                file_dir = os.path.join(subdir, file)
                filename = os.path.splitext(os.path.basename(file))[0]
                filename = filename.split('_')[-1]
                dataset_df = pd.read_csv(file_dir)
                h5_file_path = f'embedding_dataset\\{self.tlp}\\{self.domain}\\{filename}.h5'

                with h5py.File(h5_file_path, 'r') as h5_file:
                    embeddings = h5_file['embedding'][:]

                embeddings_list = [embedding for embedding in embeddings]

                dataset_df['embedding'] = embeddings_list

                df = pd.concat([df, dataset_df])

        return df
    

   
