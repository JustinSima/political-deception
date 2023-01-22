""" Pipeline that loads a pretrained Bert model and tokenizer, 
and fine-tunes the model on HuggingFace's 'liar' dataset.
Includes functionality to train, evaluate, and save model.
"""
import datetime
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm, trange

import datasets
import numpy as np
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
)


class DolosTrainer:
    def __init__(self, model_dir:str, device=None):
        # Set device if none given.
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available() 
                else 'cpu'
            )
            print(f'Device not specified. Defaulting to {self.device}')
        else:
            self.device = device

        # Point to model directory or download model.
        if not isinstance(model_dir, pathlib.Path):
            model_dir = pathlib.Path(model_dir)

        if not model_dir.exists():
            base_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

            base_model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir / 'tokenizer')

        else:
            base_model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
            self.tokenizer = BertTokenizer.from_pretrained(model_dir / 'tokenizer')

        self.model = nn.DataParallel(base_model)
        self.model.to(self.device)

        # Define trainable model parameters.
        params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.1
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0
                }
        ]

        # Initialize additional attributes.
        self.loss_train_list = None
        self.loss_val_list = None
        self.train_dataloader = None
        self.validation_dataloader = None

    def tokenize_dataset(self, name: str, batch_size: int=32):
        # Check input.
        if name not in ['train', 'validation', 'test']:
            raise ValueError(f"Variable 'name' must be one of 'train', 'validation', 'test'. Value received: {name}")

        dataset = datasets.load_dataset('liar', split=name)
        data = dataset.map(lambda sample: self.tokenizer(sample['statement'], padding='max_length', truncation=True), batched=True)
        data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label']) # 'token_type_ids',
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        attr_name = name + '_dataloader'
        setattr(self, attr_name, data_loader)

    def train(self,
        epochs: int=2, learning_rate: float=1e-5, epsilon: float=1e-8
    ):
        # Prepare optimizer and scheduler.
        optimizer = AdamW(self.optimizer_params, lr=learning_rate, eps=epsilon)
        total_steps = len(self.train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=0, num_training_steps=total_steps)

        # Training loop.
        self.loss_train_list = []
        self.loss_val_list = []

        for epoch in trange(epochs, leave=True, desc='Epoch:'):
            self.model.train()

            # Initialize epoch tracking variables.
            time_start = datetime.datetime.now()
            loss_train, accuracy_train = 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            val_loss, val_accuracy = 0, 0
            n_val_steps = 0
            
            for batch in tqdm(self.train_dataloader, leave=True, desc='Batches:'):
                # Store tensors and move to device.
                batch_sequences, batch_masks, batch_labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['label'].to(self.device)

                optimizer.zero_grad()
                
                # Feed model and calculate loss / accuracy.
                outputs = self.model(batch_sequences, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)

                loss = outputs['loss']

                self.loss_train_list.append(loss.item())
                logits = outputs['logits'].detach().cpu().numpy()
                np_labels = batch_labels.to('cpu').numpy()
                batch_train_accuracy = accuracy_score(logits, np_labels)
                accuracy_train += batch_train_accuracy
                
                # Backwards step.
                loss.backward()
                optimizer.step()
                scheduler.step()
                    
                # Update train tracking statistics.
                loss_train += loss.item()
                nb_tr_examples += batch_sequences.size(0)
                nb_tr_steps += 1

            time_elapsed = datetime.datetime.now() - time_start

            # Evaluate each epoch.
            self.model.eval()
                                
            for batch in self.validation_dataloader:
                batch_sequences, batch_masks, batch_labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['label'].to(self.device)
                                
                with torch.no_grad():
                    output = self.model(batch_sequences, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
                    logits = output['logits'].detach().cpu().numpy()
                    np_labels = batch_labels.to('cpu').numpy()
                    
                    batch_val_accuracy = accuracy_score(logits, np_labels)
                    batch_val_loss = output['loss']
                    self.loss_val_list.append(batch_val_loss.item())
                    val_loss += batch_val_loss.item()
                    val_accuracy += batch_val_accuracy
                    n_val_steps += 1
            
            len_train = len(self.train_dataloader)
            len_val = len(self.validation_dataloader)
            print(f"Epoch: {epoch}, \n\
                Average Time per Batch: {time_elapsed / len_train}, \n\
                Training Loss: {loss_train / len_train} \t\
                Training Accuracy: {accuracy_train / len_train} \n\
                Validation Loss: {val_loss / len_val} \t\
                Validation Accuracy: {val_accuracy / len_val}\n")

    def plot_training_loss(self, save_path: str):
        plt.figure(figsize=(15,8))
        plt.title("Training loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(self.loss_train_list)
        plt.savefig(save_path)
        plt.close()

    def plot_validation_loss(self, save_path: str):
        plt.figure(figsize=(15,8))
        plt.title("Validation loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(self.loss_val_list)
        plt.savefig(save_path)
        plt.close()

    def evaluate_test_performance(self, eval_fn=None):
        if eval_fn is None:
            eval_fn = matthews_corrcoef
            print("No metric function provided. Defaulting to aggregate Matthew's coefficient.")

        self.model.eval()

        preds = []
        true_state = []

        for batch in tqdm(self.test_dataloader):
            batch_sequences = batch['input_ids'].long().to(self.device)
            batch_masks = batch['attention_mask'].long().to(self.device)
            batch_labels = batch['label'].long().to(self.device)
            
            with torch.no_grad():
                output = self.model(batch_sequences, token_type_ids=None, attention_mask=batch_masks)
            
            logits = output['logits'].detach().cpu().numpy()
            np_labels = batch_labels.to('cpu').numpy()
            preds.append(logits)
            true_state.append(np_labels)

        flattened_predictions = [item for sublist in preds for item in sublist]
        flat_predictions = np.argmax(flattened_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_state for item in sublist]

        return eval_fn(flat_true_labels, flat_predictions)

    def save_model(self, directory: str):
        if not isinstance(directory, pathlib.Path):
            directory = pathlib.Path(directory)

        if not directory.exists():
            directory.mkdir()

        try:
            torch.save(self.model.module.state_dict(), directory / 'model.pt')
        except AttributeError:
            torch.save(self.model.state_dict(), directory)

        self.tokenizer.save_pretrained(directory / 'tokenizer')

def accuracy_score(preds, labels):
    class_preds = np.argmax(preds, axis=1).flatten()
    class_labels = labels.flatten()

    return np.sum(class_preds == class_labels) / len(class_labels)
