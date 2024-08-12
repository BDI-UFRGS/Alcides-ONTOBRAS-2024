import pandas as pd
import os
import numpy as np

class DLPDatasetBuilder:
    def __init__(self, file_dir: str, output_dir: str, target_classes: list) -> None:
        self.class_super_classes_folder = 'ontologies\DLP\DLP_class_superclasses'
        self.target_classes = target_classes
        self.file_name = file_dir
        self.output_dir = output_dir
        self.original_df = pd.read_csv(file_dir, delimiter=';', names=['class', 'term', 'definition'], encoding='utf-8')
        self.original_df.dropna(inplace=True)
        self.original_df.reset_index(inplace=True)
        self.disjointness_data = self._get_disjointness_data()
        self.original_df = self._group_multi_inheritance(df=self.original_df)
        self.original_df = self._drop_incosistencies(df=self.original_df)
        self._get_processed_df()

    def _explode(self, df: pd.DataFrame):
        df = df.explode('term', ignore_index=True)
        return df
    

    def _get_disjointness_data(self) -> dict:
        data = dict()

        for r, _, files in os.walk('disjoint_classes'):
            for file_name in files:
                file = open(os.path.join(r, file_name), 'r+', encoding='utf-8')
                file_name = file_name.split('.')[0]
                data[file_name] = set()

                for line in file.readlines():
                    m_line = line.strip()
                    if m_line:
                        data[file_name].add(str(m_line))

        return data
        

    def _get_processed_df(self):
        df = pd.DataFrame()        
        
        df['term'] = self.original_df['term'].apply(lambda x: set(x[1:-1].split(',')))
        df['definition'] = self.original_df['definition']

        y_classes = self._build_multilabel_classes()
        y_classes = y_classes[self.target_classes].astype(int)

        df = pd.concat([df, y_classes], axis=1)
        df = self._explode(df=df)

        df['term'] = df['term'].astype(str).apply(lambda x: x.strip())

        df = df.loc[~(df[self.target_classes] == 0).all(axis=1)]


        filename = os.path.splitext(os.path.basename(self.file_name))[0]
        filename = filename.split('_')[-1]

        output_folder = os.path.join(self.output_dir, filename.upper())

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Folder '{output_folder}' created successfully.")
        else:
            print(f"Folder '{output_folder}' already exists.")

        df.to_csv(f'{output_folder}\\{filename}.csv', index=False)

        print(f'Final dataset size: {df.shape}')


    def _is_inconsistent(self, classes: list):
        for c1 in classes:
            for c2 in classes:
                if c1 in self.disjointness_data and c2 in self.disjointness_data[c1]:
                    return True
                
        return False

    def _drop_incosistencies(self, df: pd.DataFrame):
        incosistent_idx = []
        for idx, row in df.iterrows():
            classes = list(row['class'])
            if self._is_inconsistent(classes):
                incosistent_idx.append(idx)
        df = df.drop(incosistent_idx).reset_index(drop=True)
        return df
        

    def _group_multi_inheritance(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped_df = df.groupby(['term', 'definition'])['class'].agg(list).reset_index()
        return grouped_df

    
    def _get_classes_super_classes(self):
        classes_super_classes = dict()
        classes = set()

        for r, _, files in os.walk(self.class_super_classes_folder):
            for file_name in files:
                file = open(os.path.join(r, file_name), 'r+', encoding='utf-8')
                short_file_name = file_name.split('.')[0]

                classes_super_classes[short_file_name] = []

                for line in file.readlines():
                    classes.add(line.strip())
                    classes_super_classes[short_file_name].append(line.strip())
    
        return classes, classes_super_classes
    
    def _build_multilabel_classes(self):
        classes, classes_super_classes = self._get_classes_super_classes()
        sorted_classes = sorted(classes)

        multi_label_classes = []
        for _, row in self.original_df.iterrows():
            cls = row['class']
            new_row = np.zeros(len(sorted_classes))
            for c in cls:
                super_classes = classes_super_classes[c]

                for super_class in super_classes:
                    idx = sorted_classes.index(super_class)
                    new_row[idx] = 1

            multi_label_classes.append(new_row)

        multi_label_classes = pd.DataFrame(multi_label_classes, columns=sorted_classes)
        multi_label_classes = multi_label_classes.loc[:, (multi_label_classes != 0).any(axis=0)]

        return multi_label_classes

    def _remove_less_than(self, df: pd.DataFrame, value: int):
        ignored_columns = ['term', 'definition']
        y = df[df.columns.difference(ignored_columns)]
        columns = y[y.columns[y.sum() < value]]
        df = df.drop(columns=columns, axis=1)
        return df
    
    def _labels(self, df) -> list:
        ignored_columns = ['term', 'definition']
        y = df[df.columns.difference(ignored_columns)]
        return list(y.columns)
