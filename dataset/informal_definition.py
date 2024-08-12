import numpy as np
import pandas as pd

class InformalDefinitionDataset():
    def __init__(self, df: pd.DataFrame):
        df['term'] = df['term'].str.replace(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', regex=True)
        y = df[df.columns.difference(['term', 'definition'])]
        self.X = np.asarray(df[['term', 'definition']].astype(str).agg(' is '.join, axis=1).squeeze())
        self.y = np.asarray(y.astype(int))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]