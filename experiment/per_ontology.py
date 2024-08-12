import pandas as pd
import h5py
import numpy as np
from classifier.knn import KNNClassifier
from sklearn.metrics import classification_report
import json
import os
import numpy as np

from experiment.experiment import Experiment


class PerOntologyToTLP(Experiment):
    def __init__(self, tlp: str) -> None:
        self.tlp = tlp
        ontologies = self._get_ontologies()
        self.run(ontologies)
        # self.PCA_plot(ontologies)


    def run(self, ontologies: list):
        for train_df, test_df in self._build_train_test(ontologies):
            x_train, y_train = self._split_X_y(train_df)
            x_test, y_test = self._split_X_y(test_df)

            x_extra, y_extra, x_test, y_test = self._train_test_split(x_test, y_test, 0.9)

            x_train = pd.concat([x_train, x_extra], ignore_index=True)
            y_train = np.concatenate([y_train, y_extra])



            print(np.unique(x_train['dataset']))
            print(np.unique(x_test['dataset']))

            knn = KNNClassifier(n_neighbors=11)
            knn.fit(x_train['embedding'], y_train)

            name = str(x_test['dataset'][0]).upper()
            predictions = knn.predict(x_test['embedding'])

            self.check_right_classified(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, y_pred=predictions, name=name)
            self.check_wrong_classified(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, y_pred=predictions, name=name)

            self.confusion_matrix(y_true=y_test, y_pred=predictions, name=name)
            report = classification_report(y_true=y_test, y_pred=predictions, target_names=self.labels, output_dict=True, zero_division=0)
            with open(f"log\\{test_df['dataset'][0]}.json", 'w') as fp:
                json.dump(report, fp, indent=4)


    def _build_train_test(self, ontologies: list):
        for i in range(len(ontologies)):
            train_df = pd.concat(ontologies[:i] + ontologies[i+1:])
            test_df = ontologies[i]

            yield train_df, test_df


    def _get_ontologies(self):
        ontologies = []

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

                    ontologies.append(dataset_df)

        return ontologies






class PerFileExperiment(Experiment):
    def __init__(self, tlp: str, domain: str, dataset: str) -> None:
        self.tlp = tlp
        self.domain = domain
        self.dataset = dataset
        
        df = self._get_df()
        print(f'Original size: {df.shape}')
        df = df.drop_duplicates(subset=['term', 'definition'], ignore_index=True)
        X, y = self._split_X_y(df)


        class_counts = df[self.labels].sum()

        # Display the counts
        print(class_counts)

        # df = df.drop_duplicates(subset=['term', 'definition'], ignore_index=True)
        # print(f'After dropduplicate size: {df.shape}')




        # self.run(X, y)
    
#     def run(self, X, y):
#         for i, train_X, train_y, test_X, test_y in self._stratify(X, y):
#             print(f'Performing FOLD {i} experiment in {self.tlp.upper()}-{self.dataset.upper()} ontology')

#             knn = KNNClassifier(n_neighbors=29)
#             knn.fit(train_X['embedding'], train_y)

#             predictions = knn.predict(test_X['embedding'])

#             self.confusion_matrix(y_true=test_y, y_pred=predictions, x_test=test_X)
#             report = classification_report(y_true=test_y, y_pred=predictions, target_names=self.labels, output_dict=True, zero_division=0)
#             with open(f'log\\report_{self.tlp}_{self.domain}_{self.dataset}_FOLD_{i}.json', 'w') as fp:
#                 json.dump(report, fp, indent=4)

#     def confusion_matrix(self, y_true, y_pred, x_test):
#         y_true = np.argmax(y_true, axis=1)
#         y_pred = np.argmax(y_pred, axis=1)

#         initials = [''.join([word[0] for word in label.split()]).upper() for label in self.labels]


#         y_true_labels = [initials[i] for i in y_true]
#         y_pred_labels = [initials[i] for i in y_pred]

#         cm = confusion_matrix(y_true_labels, y_pred_labels, labels=initials)

#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=initials)
#         disp.plot(cmap=plt.cm.Blues)
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()

#         df = np.array(x_test['term'] + ' is ' + x_test['definition'])

#         for i in range(len(y_true_labels)):
#             if y_true_labels[i] != y_pred_labels[i]:
#                 print(f"True: {y_true_labels[i]} Predicted: {y_pred_labels[i]} - {df[i]}")


    def _get_df(self) -> pd.DataFrame:
        df = pd.read_csv(f'input_dataset\\{self.tlp}\\{self.domain}\\{self.dataset}.csv')
        
        h5_file_path = f'embedding_dataset\\{self.tlp}\\{self.domain}\\{self.dataset}.h5'

        with h5py.File(h5_file_path, 'r') as h5_file:
            embeddings = h5_file['embedding'][:]

        embeddings_list = [embedding for embedding in embeddings]

        df['embedding'] = embeddings_list
        df['domain'] = [self.domain.upper()] * len(df)
        df['dataset'] = [self.dataset.upper()] * len(df)

        return df
    

#     def _split_X_y(self, df: pd.DataFrame):
#         X_columns = ['term', 'definition', 'embedding', 'domain', 'dataset']
#         X = df[X_columns]
#         self.labels = []

#         for column in df.columns:
#             if column not in X_columns:
#                 self.labels.append(column)

#         y = df[self.labels]

#         return X, np.asarray(y)

#     def _stratify(self, X: pd.DataFrame, y: pd.DataFrame):
#         mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         for i, (train_index, test_index) in enumerate(mlskf.split(X, y)):
#             train_X = X.loc[train_index]
#             train_y = y[train_index]

#             test_X = X.loc[test_index]
#             test_y = y[test_index]
            
#             yield i, train_X, train_y, test_X, test_y

