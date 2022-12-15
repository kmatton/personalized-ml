"""
Helper functions for reading and processing datasets.
"""
import csv
import os

import numpy as np
import pandas as pd
from IPython import embed


def read_raw_amazon_reviews_data(data_dir, raw_data_file, split_file):
    # read in data
    data_df = pd.read_csv(os.path.join(data_dir, raw_data_file),
                          dtype={'reviewerID': str, 'asin': str, 'reviewTime': str, 'unixReviewTime': int,
                                 'reviewText': str, 'summary': str, 'verified': bool, 'category': str,
                                 'reviewYear': int},
                          keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONNUMERIC)
    data_df = data_df.rename(columns={'reviewerID': 'user', 'reviewText': 'text', 'overall': 'labels'})
    data_df["sample_id"] = data_df.index
    # transform labels
    data_df["labels"] = data_df["labels"].apply(lambda x: x - 1)
    data_df["labels"] = data_df["labels"].astype('long')
    # add data splits
    split_df = pd.read_csv(os.path.join(data_dir, "splits", split_file))
    is_in_dataset = split_df["split"] != -1
    data_df = data_df[is_in_dataset]
    split_df = split_df[is_in_dataset]

    # default splits from this dataset have 0-4, including 2 - ID validation, and 4 - ID test
    five_splits = True
    if split_file != "users.csv":
        five_splits = False

    def split_int_to_str(x):
        if x == 0:
            return "train"
        elif x == 1:
            return "val"
        elif x == 2 and not five_splits:
            return "test"
        elif x == 3 and five_splits:
            return "test"
        return ""

    data_df["split"] = split_df["split"].apply(lambda x: split_int_to_str(x))
    return data_df


def filter_data(dataset, max_samples, select_random, filter_by, keep_vals):
    if max_samples is not None:
        if not select_random:
            print("taking first {} samples".format(max_samples))
            dataset = dataset.select(range(max_samples))
        else:
            print("taking random {} samples".format(max_samples))
            full_idx = np.arange(len(dataset))
            keep_idx = np.random.choice(full_idx, max_samples, replace=False)
            dataset = dataset.select(keep_idx)
    if filter_by is not None:
        print("filtering to include only samples where {} is in {}".format(filter_by, keep_vals))
        keep_vals = set(keep_vals)
        dataset = filter_helper_multi_val(dataset, filter_by, keep_vals)
    return dataset


def filter_helper_single_val(dataset, filter_by, keep_val):
    vals = dataset[filter_by]
    keep_idx = [i for i in range(len(vals)) if vals[i] == keep_val]
    return dataset.select(keep_idx)


def filter_helper_multi_val(dataset, filter_by, keep_vals):
    vals = dataset[filter_by]
    keep_idx = [i for i in range(len(vals)) if vals[i] in keep_vals]
    return dataset.select(keep_idx)


def split_data(dataset):
    train_data = dataset.filter(lambda x: x["split"] == "train")
    val_data = dataset.filter(lambda x: x["split"] == "val")
    test_data = dataset.filter(lambda x: x["split"] == "test")
    return train_data, val_data, test_data
