"""
For loading MNIST data.
"""
import os

from datasets import load_dataset
from transformers import default_data_collator
from IPython import embed

from dataset.dataset import BaseDataset


class MNISTDataset(BaseDataset):
    """
    Class for MNIST data.
    """
    _data_name = "MNIST"

    def __init__(self, data_dir, raw_data_file, processed_data_dir=None, split_file=None, user_col=None,
                 max_train_samples=None, max_val_samples=None, max_test_samples=None, save_dataset=True):
        super().__init__(data_dir, raw_data_file, processed_data_dir, split_file, user_col, max_train_samples,
                         max_val_samples, max_test_samples, save_dataset)
        max_label = int(max(self.train_data["labels"])[0])
        self.d_out = max_label
        if max_label != 1:
            self.d_out += 1

    def read_raw_data(self):
        dataset = load_dataset('json', data_files=os.path.join(self.data_dir, self.raw_data_file),
                               field="data", split="all")
        dataset = dataset.rename_column('x', "embeddings")
        dataset = dataset.add_column('labels', [[y] for y in dataset['y']])
        dataset = dataset.remove_columns("y")
        return dataset

    def get_data_collator(self, is_train):
        return default_data_collator

    def preprocess_data_split(self, dataset, data_split):
        return dataset
