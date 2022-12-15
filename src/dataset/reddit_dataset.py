import os

import pandas as pd

from dataset.mlm_dataset import MLMDataset


class RedditDataset(MLMDataset):
    _data_name = "reddit_mlm"

    def read_raw_data(self):
        # read in data
        data_df = pd.read_csv(os.path.join(self.data_dir, self.raw_data_file))
        data_df.drop(columns="Unnamed: 0", inplace=True)
        return data_df
