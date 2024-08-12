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


class CrossTLPExperiment:
    def __init__(self, train_tlp: str, test_tlp: str) -> None:
        self.train_tlp = train_tlp
        self.test_tlp = test_tlp

        train_df = self._get_df(train_tlp)
        print(f'Original size: {train_df.shape}')
        train_df = train_df.drop_duplicates(subset=['term', 'definition'], ignore_index=True)
        print(f'After dropduplicate size: {train_df.shape}')


        test_df = self._get_df(test_tlp)
        print(f'Original size: {test_df.shape}')
        test_df = test_df.drop_duplicates(subset=['term', 'definition'], ignore_index=True)
        print(f'After dropduplicate size: {test_df.shape}')

        # train_df = train_df.sample(1000, ignore_index=True)
        # test_df = test_df.sample(1000, ignore_index=True)
        
        X_train, y_train, train_labels = self._split_X_y(train_df)
        X_test, y_test, test_labels = self._split_X_y(test_df)

        self.run(X_train, y_train, train_labels, X_test, y_test, test_labels)
    

    def run(self, X_train, y_train, train_labels, X_test, y_test, test_labels):
        print(f'Performing cross TLP experiment using {self.train_tlp.upper()} to train and {self.test_tlp.upper()} to test')
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(X_train['embedding'], y_train)
        predictions = knn.predict(X_test['embedding'])
        self.plot(y_true=y_test, y_pred=predictions, train_labels=train_labels, test_labels=test_labels)
        # self.confusion_matrix(y_true=test_y, y_pred=predictions)        
        # report = classification_report(y_true=test_y, y_pred=predictions, target_names=self.labels, output_dict=True, zero_division=0)
        # with open(f'log\\report_{self.tlp}_FOLD_{i}.json', 'w') as fp:
        #     json.dump(report, fp, indent=4)


    def plot(self, y_true, y_pred, train_labels: list, test_labels: list):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        y_true_labels = [str(test_labels[i]).title() for i in y_true]
        y_pred_labels = [str(train_labels[i]).title() for i in y_pred]


        for train_label in train_labels:
            print(f'For {train_label}')
            values = dict()
            count = 0
            for i, y_pred_label in enumerate(y_pred_labels):
                if y_pred_label == str(train_label).title():
                    count += 1
                    if y_true_labels[i] not in values:
                        values[y_true_labels[i]] = 0
                    values[y_true_labels[i]] += 1

            labels = values.keys()
            sizes = values.values()

            # plt.figure(figsize=(6, 6))
            # wedges, texts = plt.pie(sizes, startangle=140)
            # plt.title(f'Distribution for {str(train_label).title()}')
            
            # # Adding legend

            display_labels = ['\n'.join(label.split()) for label in labels]

            
            # plt.show()

            # labels = values.keys()
            # sizes = values.values()

            plt.figure(figsize=(5, 5))
            wedges, texts, autotexts = plt.pie(sizes, autopct='%1.1f%%', startangle=140)
            
            # Customize the autopct text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                # autotext.set_fontsize(12)
            
            label = '\n'.join(train_label.split())
            # Adding legend
            plt.legend(wedges, display_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.title(f'{label.title()}', fontsize=10)
            plt.tight_layout()
            plt.show()

        # print(y_true_labels)
        # print(y_pred_labels)

    def _get_df(self, tlp: str) -> pd.DataFrame:

        df = pd.DataFrame()
        folder_dir = os.path.join('input_dataset', tlp)

        for subdir, _, files in os.walk(folder_dir):
            if folder_dir != subdir:
                domain = subdir.split('\\')[2]
        
                for file in files:
                    file_dir = os.path.join(subdir, file)
                    filename = os.path.splitext(os.path.basename(file))[0]
                    filename = filename.split('_')[-1]
                    
                    dataset_df = pd.read_csv(file_dir)
                    h5_file_path = f'embedding_dataset\\{tlp}\\{domain}\\{filename}.h5'

                    with h5py.File(h5_file_path, 'r') as h5_file:
                        embeddings = h5_file['embedding'][:]

                    embeddings_list = [embedding for embedding in embeddings]

                    dataset_df['embedding'] = embeddings_list
                    dataset_df['domain'] = [domain.upper()] * len(dataset_df)
                    dataset_df['dataset'] = [filename.upper()] * len(dataset_df)

                    df = pd.concat([df, dataset_df])

        return df
    

    def _split_X_y(self, df: pd.DataFrame):
        X_columns = ['term', 'definition', 'embedding', 'domain', 'dataset']
        X = df[X_columns]
        labels = []

        for column in df.columns:
            if column not in X_columns:
                labels.append(column)

        y = df[labels]

        return X, np.asarray(y), labels

    def _stratify(self, X: pd.DataFrame, y: pd.DataFrame):
        mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(mlskf.split(X, y)):
            train_X = X.loc[train_index]
            train_y = y[train_index]

            test_X = X.loc[test_index]
            test_y = y[test_index]
            
            yield i, train_X, train_y, test_X, test_y

    def confusion_matrix(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        initials = [''.join([word[0] for word in label.split()]).upper() for label in self.labels]


        y_true_labels = [initials[i] for i in y_true]
        y_pred_labels = [initials[i] for i in y_pred]

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=initials)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=initials)
        disp.plot(cmap=plt.cm.Blues)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        df = np.array(self.df['term'] + ' is ' + self.df['definition'])
        domain = np.array(self.df['domain'])
        dataset = np.array(self.df['dataset'])

        for i in range(len(y_true_labels)):
            if y_true_labels[i] != y_pred_labels[i]:
                print(f"Domain: {domain[i]} Dataset: {dataset[i]} True: {y_true_labels[i]} Predicted: {y_pred_labels[i]} - {df[i]}")

