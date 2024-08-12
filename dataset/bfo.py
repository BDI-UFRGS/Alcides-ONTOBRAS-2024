import pandas as pd
import os
import numpy as np
import spacy

# Load a pre-trained language model
nlp = spacy.load('en_core_web_sm')

class BFODatasetBuilder:
    def __init__(self, folder_dir: str, output_dir: str, target_classes: list) -> None:
        self.target_classes = target_classes

        for subdir, _, files in os.walk(folder_dir):
            if folder_dir != subdir:
                domain = subdir.split('\\')[2]
                output_folder = os.path.join(output_dir, domain)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    print(f"Folder '{output_folder}' created successfully.")
                else:
                    print(f"Folder '{output_folder}' already exists.")

                for file in files:
                    file_dir = os.path.join(subdir, file)
                    filename = os.path.splitext(os.path.basename(file))[0]
                    filename = filename.split('_')[-1]
                    df = self._extract_dataset(file_dir)

                    df.to_csv(f'{output_folder}\\{filename}.csv', index=False)


    def _extract_dataset(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, delimiter=';', names=['class', 'term', 'definition'], encoding='utf-8')
        df = self._filter_definitions(df=df)
        classes = self._one_hot_encoding(df=df)     

        output_df = pd.DataFrame()
        output_df['term'] = df['term']
        output_df['definition'] = df['definition']
        # output_df['if'] = df['term'] + ' is ' + df['definition']
        # output_df['genus'] = output_df['if'].apply(self._extract_genus)
        output_df = pd.concat([output_df, classes], axis=1)

        return output_df
        

    def _one_hot_encoding(self, df: pd.DataFrame):
        # Convert the 'class' column to a categorical type with all possible classes
        df['class'] = pd.Categorical(df['class'], categories=self.target_classes)

        # df['type'] = df['class'].apply(lambda x: 'occurrent' if x in ['process'] else 'continuant')
        # df['type'] = pd.Categorical(df['type'], categories=['continuant', 'occurrent'])

        # print(df['type'])

        # Use get_dummies to create one-hot encoding for the 'class' column
        one_hot_encoded = pd.get_dummies(df['class'], columns=['class'])
        # one_hot_encoded2 = pd.get_dummies(df['type'], columns=['type'])

        # one_hot_encoded = pd.concat([one_hot_encoded1, one_hot_encoded2], axis=1)

        return one_hot_encoded.astype(int)


    def _extract_genus(self, definition):
        doc = nlp(definition)
        copula_index = None
        
        # Find the copula
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "AUX":
                copula_index = token.i
                break
        
        if copula_index is None:
            return None
        
        # Find the genus phrase (usually a noun phrase after the copula)
        genus_phrase = []
        for token in doc[copula_index + 1:]:
            if token.pos_ in ["NOUN", "ADJ", "DET"]:
                genus_phrase.append(token.text)
            elif genus_phrase:
                break

        return " ".join(genus_phrase)

    def _filter_definitions(self, df: pd.DataFrame):
        df['definition'] = df['definition'].astype(str)

        mask = df['definition'].apply(lambda x: len(x.split())) >= 5
        filtered_df = df[mask]

        return filtered_df
    

