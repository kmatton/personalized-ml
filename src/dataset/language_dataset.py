"""
Abstract Dataset Class for language data.
"""
import os

import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, default_data_collator
from IPython import embed

from dataset.dataset import BaseDataset


class LanguageDataset(BaseDataset):
    """
    Shared dataset class for all language datasets
    """
    _data_name = "generic_language"

    def __init__(self, data_dir, raw_data_file, processed_data_dir=None, split_file=None, user_col=None,
                 max_train_samples=None, max_val_samples=None, max_test_samples=None, save_dataset=True):
        super().__init__(data_dir, raw_data_file, processed_data_dir, split_file, user_col,
                         max_train_samples, max_val_samples, max_test_samples, save_dataset)

    def preprocess_data_split(self, dataset, data_split):
        dataset = self._tokenize_data(dataset, data_split)
        return dataset

    def embed_data(self, embed_model, embed_path):
        def _embed_batch(input_ids, attention_mask):
            output = embed_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_state = output.hidden_states[-1][:, 0]
            return hidden_state

        # check if embeddings are stored in single file
        embed_file = os.path.join(embed_path, "embeddings.npy")
        if os.path.exists(embed_file):
            print("loading embeddings for all data at path {}".format(embed_file))
            embeds = np.load(embed_file, allow_pickle=True)
            ids_file = os.path.join(embed_path, "sample_ids.npy")
            assert os.path.exists(ids_file), "need to provide sample ids file when providing embeddings file for full dataset"
            sample_ids = np.load(ids_file, allow_pickle=True)
            self._add_precomputed_embeddings(sample_ids, embeds)
        else:
            self.train_data = self._get_embedded_data(embed_path, _embed_batch, self.train_data, "train")
            self.val_data = self._get_embedded_data(embed_path, _embed_batch, self.val_data, "val")
            self.test_data = self._get_embedded_data(embed_path, _embed_batch, self.test_data, "test")
        self.embed_dim = len(self.train_data["embeddings"][0])

    def _tokenize_data(self, dataset, data_split):
        print("tokenizing {} data".format(data_split))
        # ignore data split because process all splits the same
        dataset = dataset.map(
            self._tokenize_func,
            batched=True,
            num_proc=self.preprocessing_num_workers
        )
        return dataset

    def _tokenize_func(self, examples):
        raise NotImplementedError

    def _add_precomputed_embeddings(self, sample_ids, embeds):
        # add embeddings to datasets
        train_ids = self.train_data["sample_id"]
        train_embeds = self._add_precomputed_embeddings_helper(sample_ids, train_ids, embeds)
        self.train_data = self.train_data.add_column(name="embeddings", column=train_embeds.tolist())
        val_ids = self.val_data["sample_id"]
        val_embeds = self._add_precomputed_embeddings_helper(sample_ids, val_ids, embeds)
        self.val_data = self.val_data.add_column(name="embeddings", column=val_embeds.tolist())
        test_ids = self.test_data["sample_id"]
        test_embeds = self._add_precomputed_embeddings_helper(sample_ids, test_ids, embeds)
        self.test_data = self.test_data.add_column(name="embeddings", column=test_embeds.tolist())

    def _add_precomputed_embeddings_helper(self, all_sample_ids, split_sample_ids, embeds):
        embeds_idx = np.array([np.argwhere(all_sample_ids == x) for x in split_sample_ids]).flatten()
        split_embeds = embeds[embeds_idx]
        return split_embeds

    def _get_embedded_data(self, embed_path, embed_fn, split_dataset, data_split):
        full_path = None
        embeds_dir = os.path.join(embed_path, self.get_data_split_name(data_split))
        if embed_path is not None:
            full_path = os.path.join(embeds_dir, "embeddings.npy")
        if os.path.exists(full_path):
            print('loading existing embeddings for {} data'.format(data_split))
            embeds = np.load(full_path, allow_pickle=True)
        else:
            embeds = self._embed_data_split(embed_fn, split_dataset, data_split)
            if full_path is not None:
                print('saving embeddings to {}'.format(full_path))
                if not os.path.exists(embeds_dir):
                    os.makedirs(embeds_dir)
                np.save(full_path, embeds)
        split_dataset = split_dataset.add_column(name="embeddings", column=embeds.tolist())
        return split_dataset

    def _embed_data_split(self, embed_fn, split_dataset, data_split):
        print("embedding {} data".format(data_split))
        embeds = None
        data_loader = self._get_embed_dataloader(split_dataset)
        for batch in tqdm(data_loader):
            embedded_batch = embed_fn(batch['input_ids'], batch['attention_mask'])
            embedded_batch = embedded_batch.detach().cpu().numpy()
            embeds = embedded_batch if embeds is None else np.concatenate((embeds, embedded_batch), axis=0)
        return embeds

    def _get_embed_dataloader(self, split_dataset):
        sampler = SequentialSampler(split_dataset)
        data_loader = DataLoader(
            split_dataset,
            batch_size=8,
            sampler=sampler,
            collate_fn=default_data_collator,
            drop_last=False
        )
        return data_loader

    def load_tokenizer(self):
        tokenizer_kwargs = {
            "use_fast": True,
            "cache_dir": self.tokenizer_cache_dir
        }
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, **tokenizer_kwargs)
        return tokenizer
