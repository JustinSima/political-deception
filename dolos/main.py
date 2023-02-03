""" Run training pipeline for Dolos model and save.
"""
import pathlib

from model import Dolos
from training import DolosTrainer


# Point to pretrained Bert model.
model_dir = pathlib.Path('../../../') / 'data' / 'models' / 'bert-base-cased'
output_dir = pathlib.Path('../../../') / 'data' / 'models' / 'dolos-finetuned'
image_dir = pathlib.Path('../../../') / 'data' / 'images' / 'experiment-1'

# Fine-tune model
trainer = DolosTrainer(model_dir=model_dir)
trainer.tokenize_dataset('train')
trainer.tokenize_dataset('validation')

trainer.train()
trainer.plot_training_loss(save_path=image_dir / 'train-loss.png')
trainer.plot_validation_loss(save_path=image_dir / 'val-loss.png')

trainer.tokenize_dataset('test')
test_score = trainer.evaluate_test_performance()

trainer.save_model(directory=output_dir)
