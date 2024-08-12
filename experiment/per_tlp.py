import pandas as pd
import h5py
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
from classifier.knn import KNNClassifier
from sklearn.metrics import classification_report
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from experiment.experiment import Experiment

class PerTLPExperiment(Experiment):
    def __init__(self, tlp: str) -> None:
        self.tlp = tlp

        df = self._get_df()
        print(f'Original size: {df.shape}')
        df = df.drop_duplicates(subset=['term', 'definition'], ignore_index=True)

        # df = df.sample(1000, ignore_index=True)
        print(f'After dropduplicate size: {df.shape}')
        X, y = self._split_X_y(df)

        # class_counts = df[self.labels].sum()

        # # Display the counts
        # print(class_counts)


        self.run(X, y)
    
    def run(self, X, y):
        x_train, x_test, y_train, y_test = self._train_test_split(X, y)

        # for i, train_X, train_y, test_X, test_y in self._stratify(X, y):
        print(f'Performing experiment in {self.tlp.upper()} top-level ontology')
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(x_train['embedding'], y_train)
        predictions = knn.predict(x_test['embedding'])

        self.check_right_classified(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, y_pred=predictions, name='bfo')
        self.check_wrong_classified(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, y_pred=predictions, name='bfo')

        # self.confusion_matrix(y_true=y_test, y_pred=predictions)        
        # report = classification_report(y_true=y_test, y_pred=predictions, target_names=self.labels, output_dict=True, zero_division=0)
        # with open(f'log\\report_{self.tlp}.json', 'w') as fp:
        #     json.dump(report, fp, indent=4)

    
    def _get_df(self) -> pd.DataFrame:

        df = pd.DataFrame()
        folder_dir = os.path.join('input_dataset', self.tlp)

        for subdir, _, files in os.walk(folder_dir):
            if folder_dir != subdir:
                domain = subdir.split('\\')[2]
        
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
    

    

