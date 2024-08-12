import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.model_selection import iterative_train_test_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import json

class Experiment:
    def __init__(self) -> None:
        pass

    def confusion_matrix(self, y_true, y_pred, name: str):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        # initials = [''.join([word[0] for word in label.split()]).upper() for label in self.labels]

        y_true_labels = [self.labels[i] for i in y_true]
        y_pred_labels = [self.labels[i] for i in y_pred]

        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.labels)

        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized[np.isnan(cm_normalized)] = 0  # Set NaNs to zero

        # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        display_labels = ['\n'.join(label.split()) for label in self.labels]

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues)
        disp.im_.set_clim(0, 1)

        plt.xticks(rotation=45)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f'confusion matrix\\{name.upper()}_cm.png', bbox_inches='tight')


    def check_right_classified(self, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test, y_pred, name: str):
        # Initialize KNN with 5 neighbors
        X_train = X_train.reset_index()

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        y_train_labels = np.array([self.labels[i] for i in y_train])
        y_test_labels =  np.array([self.labels[i] for i in y_test])
        y_pred_labels =  np.array([self.labels[i] for i in y_pred])
        
        evaluation_domain = dict()
        evaluation_ontology = dict()

        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(list(X_train['embedding']))
        print(X_train)
        # Find the 5 closest training instances for each test sample

        # for x_test in X_test:
        distances, indices = knn.kneighbors(list(X_test['embedding']))

        # # Check if the majority class of the 5 neighbors matches the predicted class
        # correct_count = 0
        # total_count = len(y_test)
        
        for i, neighbors in enumerate(indices):
            
            if y_test[i] == y_pred[i]:
                neighbor_classes = pd.DataFrame()
                neighbor_classes['informal definition'] = X_train.loc[neighbors]['term'] + ' is ' + X_train.loc[neighbors]['definition']
                neighbor_classes['dataset'] = X_train.loc[neighbors]['dataset']
                neighbor_classes['domain'] = X_train.loc[neighbors]['domain']
                neighbor_classes['class'] = y_train_labels[neighbors]

                for _, neighbor_class in neighbor_classes.iterrows():
                    if neighbor_class['class'] == y_test_labels[i]:
                        if neighbor_class['domain'] not in evaluation_domain:
                            evaluation_domain[neighbor_class['domain']] = 0
                        evaluation_domain[neighbor_class['domain']] += 1

                        if neighbor_class['dataset'] not in evaluation_ontology:
                            evaluation_ontology[neighbor_class['dataset']] = 0
                        evaluation_ontology[neighbor_class['dataset']] += 1

        with open(f'log\\right_domain_{name}.json', 'w') as fp:
            json.dump(evaluation_domain, fp, indent=4)           
            
        with open(f'log\\right_ontology_{name}.json', 'w') as fp:
            json.dump(evaluation_ontology, fp, indent=4)           
                

        

    def check_wrong_classified(self, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test, y_pred, name: str):
        # Initialize KNN with 5 neighbors
        X_train = X_train.reset_index()

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        y_train_labels = np.array([self.labels[i] for i in y_train])
        y_test_labels =  np.array([self.labels[i] for i in y_test])
        y_pred_labels =  np.array([self.labels[i] for i in y_pred])
        
        evaluation_domain = dict()
        evaluation_ontology = dict()

        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(list(X_train['embedding']))
        print(X_train)
        # Find the 5 closest training instances for each test sample

        # for x_test in X_test:
        distances, indices = knn.kneighbors(list(X_test['embedding']))

        # # Check if the majority class of the 5 neighbors matches the predicted class
        # correct_count = 0
        # total_count = len(y_test)
        
        for i, neighbors in enumerate(indices):
            
            if y_test[i] != y_pred[i]:
                neighbor_classes = pd.DataFrame()
                neighbor_classes['informal definition'] = X_train.loc[neighbors]['term'] + ' is ' + X_train.loc[neighbors]['definition']
                neighbor_classes['dataset'] = X_train.loc[neighbors]['dataset']
                neighbor_classes['domain'] = X_train.loc[neighbors]['domain']
                neighbor_classes['class'] = y_train_labels[neighbors]

                for _, neighbor_class in neighbor_classes.iterrows():
                    if neighbor_class['class'] != y_test_labels[i]:
                        if neighbor_class['domain'] not in evaluation_domain:
                            evaluation_domain[neighbor_class['domain']] = 0
                        evaluation_domain[neighbor_class['domain']] += 1

                        if neighbor_class['dataset'] not in evaluation_ontology:
                            evaluation_ontology[neighbor_class['dataset']] = 0
                        evaluation_ontology[neighbor_class['dataset']] += 1

        with open(f'log\\wrong_domain_{name}.json', 'w') as fp:
            json.dump(evaluation_domain, fp, indent=4)           
            
        with open(f'log\\wrong_ontology_{name}.json', 'w') as fp:
            json.dump(evaluation_ontology, fp, indent=4) 


    def _split_X_y(self, df: pd.DataFrame):
        X_columns = ['term', 'definition', 'embedding', 'domain', 'dataset']
        X = df[X_columns]
        self.labels = []

        for column in df.columns:
            if column not in X_columns:
                self.labels.append(column)

        y = df[self.labels]

        return X, np.asarray(y)


    def _train_test_split(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float):
        
        columns = X.columns

        X_train, y_train, X_test, y_test = iterative_train_test_split(X.values, y, test_size=test_size)
        
        return pd.DataFrame(X_train, columns=columns), y_train, pd.DataFrame(X_test, columns=columns), y_test 
    


    
    def _stratify(self, X: pd.DataFrame, y: pd.DataFrame):
        mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(mlskf.split(X, y)):
            train_X = X.loc[train_index]
            train_y = y[train_index]

            test_X = X.loc[test_index]
            test_y = y[test_index]
            
            yield i, train_X, train_y, test_X, test_y 

        # df = np.array(x_test['term'] + ' is ' + x_test['definition'])
        # domain = np.array(x_test['domain'])
        # dataset = np.array(x_test['dataset'])

        # for i in range(len(y_true_labels)):
        #     if y_true_labels[i] != y_pred_labels[i]:
        #         print(f"Domain: {domain[i]} Dataset: {dataset[i]} True: {y_true_labels[i]} Predicted: {y_pred_labels[i]} - {df[i]}")


    def PCA_plot(self, ontologies):
        # ontology = ontologies[:3]

        colors = plt.cm.get_cmap('tab20').colors
        markers = ['o', 's', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        combinations = list(itertools.product(colors, markers))


        for idx, df in enumerate(ontologies):
            # Apply PCA to reduce to 2 dimensions
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(list(df['embedding']))
            
            # Create a DataFrame for easier plotting
            reduced_df = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2'])
            
            color, marker = combinations[idx % len(combinations)]

            # Plot the reduced dimensions with distinct colors
            plt.scatter(reduced_df['PC1'], reduced_df['PC2'], color=color, marker=marker, label=f'Dataframe {idx + 1}', edgecolor='k', s=40)
