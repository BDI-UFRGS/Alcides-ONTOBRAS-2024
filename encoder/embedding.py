from encoder.bert import BERTEmbedding
from dataset.informal_definition import InformalDefinitionDataset
import pandas as pd
from tqdm import tqdm
import h5py
import glob
import os

class EmbeddingGenerator:
    def __init__(self, input_dir: str, output_dir: str) -> None:

        encoder = BERTEmbedding(tokenizer_name='google-bert/bert-base-uncased', model_name='google-bert/bert-base-uncased')
        # encoder = BERTEmbedding(tokenizer_name='dmis-lab/biobert-v1.1', model_name='dmis-lab/biobert-v1.1')

        for subdir, _, files in os.walk(input_dir):
            if input_dir != subdir:
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
                    
                    df = pd.read_csv(file_dir, header=0)
                    
                    encodding_dataset = InformalDefinitionDataset(df)

                    embeddings = []
                    print(f'Creating embedding for: {filename.upper()}')
                    for X, _ in tqdm(encodding_dataset):
                        embeddings.append(encoder.encode(X))

                    with h5py.File(f'{output_folder}\\{filename}.h5', 'w') as f:
                        f.create_dataset('embedding', data=embeddings)



