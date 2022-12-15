from dataset.dataset_utils import read_raw_amazon_reviews_data
from dataset.mlm_dataset import MLMDataset


class AmazonMLMDataset(MLMDataset):
    _data_name = "amazon_reviews_mlm"

    def read_raw_data(self):
        data_df = read_raw_amazon_reviews_data(self.data_dir, self.raw_data_file, self.split_file)
        # rename labels columns because is no longer the labels
        data_df = data_df.rename(columns={'labels': 'rating'})
        return data_df
