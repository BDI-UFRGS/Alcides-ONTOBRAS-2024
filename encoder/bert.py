import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer

class BERTEmbedding:
    def __init__(self, tokenizer_name: str, model_name: str) -> None:
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_model()

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    
    def encode(self, text):
        tokenized_text = self.tokenizer(text, add_special_tokens=True,
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask=True, 
                                            return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokenized_text)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1)[0] 
            return np.asarray(sentence_embedding.cpu())

    
