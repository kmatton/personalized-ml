import torch
from transformers import DataCollatorForLanguageModeling, default_data_collator

from dataset.language_dataset import LanguageDataset


class MLMDataset(LanguageDataset):
    _data_name = "generic_mlm"

    def __init__(self, data_dir, raw_data_dir, tokenizer_name, tokenizer_cache_dir, preprocessing_num_workers=None,
                 processed_data_dir=None, split_file=None, max_train_samples=None, max_val_samples=None,
                 max_test_samples=None, save_dataset=True, fix_train_labels=False, fix_val_labels=False,
                 fix_test_labels=True, mlm_probability=0.15):
        self.tokenizer_name = tokenizer_name
        self.tokenizer_cache_dir = tokenizer_cache_dir
        self.preprocessing_num_workers = preprocessing_num_workers
        self.fix_train_labels = fix_train_labels
        self.fix_val_labels = fix_val_labels
        self.fix_test_labels = fix_test_labels
        self.mlm_probability = mlm_probability
        self.tokenizer = self.load_tokenizer()
        super().__init__(data_dir, raw_data_dir, processed_data_dir, split_file, max_train_samples, max_val_samples,
                         max_test_samples, save_dataset)
        self.d_out = self.tokenizer.vocab_size

    def read_raw_data(self):
        raise NotImplementedError

    def get_data_collator(self, is_train):
        if is_train:
            fixed_labels = self._check_fix_labels("train")
        else:
            fixed_labels = self._check_fix_labels("test")
        if fixed_labels:
            return default_data_collator
        # other wise use DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability
            )
        return data_collator

    def check_init(self):
        if (self.fix_val_labels and not self.fix_train_labels) \
                or (self.fix_train_labels and not self.fix_val_labels):
            print("ERROR: fix_train_labels and fix_val_labels must either both be true or both be false. "
                  "This is because the model trainer uses a single data collator for both training and validation."
                  "The data collator is different depending on whether the labels are fixed (used default collator)"
                  "or they are dynamically assigned during training (use data collator for LM).")
            print("Exiting...")
            exit(1)
        super().check_init()

    def get_data_name(self, split):
        suffix = ""
        if self._check_fix_labels(split):
            suffix = "_fixed_labels"
        return split + suffix

    def preprocess_data_split(self, dataset, data_split):
        fix_labels = self._check_fix_labels(data_split)
        if fix_labels:
            print("Preparing {} dataset with fixed labels".format(data_split))
            dataset = self._add_mlm_labels(dataset)
        else:
            dataset = self._tokenize_data(dataset, data_split)
        return dataset

    def _tokenize_func(self, examples):
        return self.tokenizer(
            examples["text"],
            padding=False,  # do dynamic padding to longest sequence in batch later
            truncation=True,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True
        )

    def _check_fix_labels(self, split):
        if (split == "train" and self.fix_train_labels) or (split == "val" and self.fix_val_labels) or \
                (split == "test" and self.fix_test_labels):
            return True
        return False

    def _add_mlm_labels(self, dataset):
        # tokenize data
        dataset = dataset.map(
            lambda x: self.tokenizer(x["text"], padding='max_length', truncation=True, return_special_tokens_mask=True),
            batched=True,
            num_proc=self.preprocessing_num_workers
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )
        inputs = torch.tensor(dataset['input_ids'])
        special_tokens_mask = torch.tensor(dataset['special_tokens_mask'])
        inputs, labels = collator.mask_tokens(inputs, special_tokens_mask)
        dataset = dataset.rename_column('input_ids', 'original_input_ids')
        dataset = dataset.add_column(name='input_ids', column=inputs.tolist())
        if 'labels' in dataset.features:
            # rename it because it will be replaced
            dataset = dataset.rename_column('labels', 'original_labels')
        dataset = dataset.add_column(name='labels', column=labels.tolist())
        return dataset
