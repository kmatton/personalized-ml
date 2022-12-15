from transformers import default_data_collator

from dataset.dataset_utils import read_raw_amazon_reviews_data
from dataset.language_dataset import LanguageDataset


class AmazonClfDataset(LanguageDataset):
    _data_name = "amazon_reviews_clf"

    def __init__(self, data_dir, raw_data_file, tokenizer_name, tokenizer_cache_dir, preprocessing_num_workers=None,
                 processed_data_dir=None, split_file=None, user_col=None, max_train_samples=None, max_val_samples=None,
                 max_test_samples=None, save_dataset=True):
        self.tokenizer_name = tokenizer_name
        self.tokenizer_cache_dir = tokenizer_cache_dir
        self.preprocessing_num_workers = preprocessing_num_workers
        self.tokenizer = self.load_tokenizer()
        super().__init__(data_dir, raw_data_file, processed_data_dir, split_file, user_col, max_train_samples,
                         max_val_samples, max_test_samples, save_dataset)
        self.d_out = 5  # 5 possible output classes

    def get_data_collator(self, is_train):
        return default_data_collator

    def read_raw_data(self):
        return read_raw_amazon_reviews_data(self.data_dir, self.raw_data_file, self.split_file)

    def _tokenize_func(self, examples):
        return self.tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=512
        )
