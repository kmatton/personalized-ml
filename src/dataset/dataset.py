"""
Abstract Dataset class.

Wraps around HuggingFace Dataset
"""
import os

from IPython import embed
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets

from dataset.dataset_utils import filter_data, split_data


class BaseDataset:
    """
    Shared dataset class for all dataset.
    """
    _data_name = "generic"

    def __init__(self, data_dir, raw_data_file, processed_data_dir=None, split_file=None, user_col=None,
                 max_train_samples=None, max_val_samples=None, max_test_samples=None, save_dataset=True):
        self.data_dir = data_dir
        self.processed_data_dir = processed_data_dir
        self.raw_data_file = raw_data_file
        self.split_file = split_file
        self.user_col = user_col
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.save_dataset = save_dataset
        self.train_data_name = self.get_data_split_name("train")
        self.val_data_name = self.get_data_split_name("val")
        self.test_data_name = self.get_data_split_name("test")
        self.d_in = None  # overwrite in child classes!
        self.d_out = None  # overwrite this in child classes!!
        self.train_data, self.val_data, self.test_data = self.load_data()
        self.users, self.train_users, self.val_users, self.test_users = None, None, None, None
        self.user_to_id = None
        self.num_users = None
        if self.user_col is not None:
            # get all users
            self.train_users = sorted(list(set(set(self.train_data[self.user_col]))))
            self.val_users = sorted(list(set(set(self.val_data[self.user_col]))))
            self.test_users = sorted(list(set(set(self.test_data[self.user_col]))))
            self.users = set(self.train_users).union(set(self.val_users)).union(set(self.test_users))
            self.users = sorted(list(self.users))
            self.num_users = len(self.users)
            print("found {} unique users in dataset".format(self.num_users))
            self.user_ids = np.arange(len(self.users))
            # add user id column
            self.user_to_id = {user: user_id for user, user_id in zip(self.users, self.user_ids)}
        self.user_embeddings = None
        # add sample index columns to each dataset
        if "sample_index" not in self.train_data.features:
            self.train_data = self.train_data.add_column("sample_index", np.arange(len(self.train_data)))
        if "sample_index" not in self.val_data.features:
            self.val_data = self.val_data.add_column("sample_index", np.arange(len(self.val_data)))
        if "sample_index" not in self.test_data.features:
            self.test_data = self.test_data.add_column("sample_index", np.arange(len(self.test_data)))
        # where to save data to after pre-processing
        self.save_dir = os.path.join(self.data_dir, self._data_name + "_processed")
        if save_dataset and not self._check_processed_data_exits():
            self.save_data(self.save_dir)
        # add embed dim if it exists
        self.embed_dim = None
        if "embeddings" in self.train_data.features:
            self.embed_dim = len(self.train_data["embeddings"][0])

    def get_data_collator(self, is_train):
        raise NotImplementedError

    def get_data_split_name(self, split):
        data_name = split
        if self.split_file is not None:
            suffix = "_" + self.split_file[:-4] + "_split"
            data_name += suffix
        return data_name

    def load_data(self):
        if self.processed_data_dir is not None and self._check_processed_data_exits():
            print('loading processed data from {}'.format(os.path.join(self.data_dir, self.processed_data_dir)))
            train_data = load_from_disk(os.path.join(self.data_dir, self.processed_data_dir, self.train_data_name))
            val_data = load_from_disk(os.path.join(self.data_dir, self.processed_data_dir, self.val_data_name))
            test_data = load_from_disk(os.path.join(self.data_dir, self.processed_data_dir, self.test_data_name))
        # otherwise reload
        else:
            full_dataset = self.read_raw_data()
            if isinstance(full_dataset, pd.DataFrame):
                full_dataset = Dataset.from_pandas(full_dataset)
            train_data, val_data, test_data = split_data(full_dataset)
            # apply basic preprocessing
            train_data = self.preprocess_data_split(train_data, "train")
            val_data = self.preprocess_data_split(val_data, "val")
            test_data = self.preprocess_data_split(test_data, "test")
        return train_data, val_data, test_data

    def cluster_data(self, cluster_model, cluster_path, cluster_train_data):
        self.train_data = self._get_cluster_assignments(cluster_model, cluster_path,
                                                        cluster_train_data, self.train_data, "train")
        self.val_data = self._get_cluster_assignments(cluster_model, cluster_path,
                                                      cluster_train_data, self.val_data, "val")
        self.test_data = self._get_cluster_assignments(cluster_model, cluster_path,
                                                       cluster_train_data, self.test_data, "test")

    def _get_cluster_assignments(self, cluster_model, cluster_path, cluster_train_data, split_dataset, data_split):
        full_path = None
        cluster_dir = os.path.join(cluster_path, self.get_data_split_name(data_split))
        if cluster_path is not None:
            full_path = os.path.join(cluster_dir, "clusters.npy")
        if os.path.exists(full_path):
            print("loading existing cluster assignments for {} data".format(data_split))
            preds = np.load(full_path, allow_pickle=True)
        else:
            print("getting cluster model predictions for {} data".format(data_split))
            if cluster_model.model is None:  # need to train model
                train_data = self.train_data
                if cluster_train_data == "all":
                    train_data = self.merge_data_splits()
                cluster_model.learn_clusters(train_data)
            preds = cluster_model.predict_clusters(split_dataset)
            if full_path is not None:
                print("saving assignments to {}".format(full_path))
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                np.save(full_path, preds)
        split_dataset = split_dataset.add_column(name="{}_cluster".format(cluster_model.cluster_feature),
                                                 column=preds.tolist())
        return split_dataset

    def save_data(self, save_path):
        print("saving dataset to {}".format(save_path))
        train_path = os.path.join(save_path, self.train_data_name)
        if os.path.exists(train_path):
            print('deleting and replacing existing train data located at {}'.format(train_path))
            os.remove(train_path)
        self.train_data.save_to_disk(train_path)
        val_path = os.path.join(save_path, self.val_data_name)
        if os.path.exists(val_path):
            print('deleting and replacing existing val data located at {}'.format(val_path))
            os.remove(val_path)
        self.val_data.save_to_disk(val_path)
        test_path = os.path.join(save_path, self.test_data_name)
        if os.path.exists(test_path):
            print('deleting and replacing existing test data located at {}'.format(test_path))
            os.remove(test_path)
        self.test_data.save_to_disk(test_path)

    def filter_data(self, max_train_samples, select_random_train, filter_train_by, train_keep_vals, max_val_samples,
                    select_random_val, filter_val_by, val_keep_vals, max_test_samples, select_random_test,
                    filter_test_by, test_keep_vals):
        print('filtering train data')
        self.train_data = filter_data(self.train_data, max_train_samples, select_random_train,
                                      filter_train_by, train_keep_vals)
        print('filtering val data')
        self.val_data = filter_data(self.val_data, max_val_samples, select_random_val, filter_val_by, val_keep_vals)
        print('filtering test data')
        self.test_data = filter_data(self.test_data, max_test_samples, select_random_test,
                                     filter_test_by, test_keep_vals)

    def report_metrics(self, discrete_columns=None, cont_columns=None):
        for split_name, dataset in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            print("Reporting metrics for {} data".format(split_name))
            print("Number of samples {}".format(len(dataset)))
            for col in discrete_columns:
                vals, counts = np.unique(dataset[col], return_counts=True)
                print("Distribution of col {}".format(col))
                dist_str = ""
                for val, count in zip(vals, counts):
                    dist_str += "{}: {}, ".format(val, count)
                print(dist_str)
            for col in cont_columns:
                print("Stats for col {}".format(col))
                vals = dataset[col]
                print("Mean {} Std {} Max {} Min {} Median {}".format(np.mean(vals), np.std(vals),
                                                                      max(vals), min(vals), np.median(vals)))

    def prepare_for_personalization(self, personalize_target, personalize_strategy, personalize_by):
        print("preparing data for {} personalization".format(personalize_strategy))
        if personalize_strategy == "ID-train":
            print("filtering training data to only include examples where {} = {}".format(personalize_by,
                                                                                          personalize_target))
            self.train_data = self.train_data.filter(lambda x: x[personalize_by] == personalize_target)
        if personalize_target is not None:
            print("filtering val and test data to only include examples where {} = {}".format(personalize_by,
                                                                                              personalize_target))
            self.val_data = self.val_data.filter(lambda x: x[personalize_by] == personalize_target)
            self.test_data = self.test_data.filter(lambda x: x[personalize_by] == personalize_target)

    def add_user_embeddings(self, method="one_hot"):
        if method == "one_hot":
            self.user_embeddings = np.eye(len(self.users))
        else:
            print("Unrecognized user embed method {}".format(method))
            print("Exiting...")
            exit(1)

    def read_raw_data(self):
        raise NotImplementedError

    def preprocess_data_split(self, dataset, data_split):
        raise NotImplementedError

    def merge_data_splits(self):
        dataset = concatenate_datasets([self.train_data, self.val_data, self.test_data])
        return dataset

    def _check_processed_data_exits(self):
        if not os.path.exists(os.path.join(self.data_dir, self.processed_data_dir, self.train_data_name)):
            return False
        if not os.path.exists(os.path.join(self.data_dir, self.processed_data_dir, self.val_data_name)):
            return False
        if not os.path.exists(os.path.join(self.data_dir, self.processed_data_dir, self.test_data_name)):
            return False
        return True

