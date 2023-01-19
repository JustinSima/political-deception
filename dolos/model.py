""" Load and make predictions with the Dolos model.
Model is loaded as a BertSequenceClassifier.
"""
import datetime
import pathlib
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer


class Dolos:
    def __init__(self, model_dir: str):
        # Convert to path.
        if not isinstance(model_dir, pathlib.Path):
            model_dir = pathlib.Path(model_dir)

        self.model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2, state_dict=torch.load(model_dir/'model.pt'))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir / 'tokenizer')

    def predict(self, sentences):
        pass
