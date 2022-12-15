"""
Script with supporting functions for general data analysis
(on top of raw data not outputs of experiments).
"""
import csv
import os

import numpy as np
import pandas as pd


def load_amazon_data(data_dir="/data/ddmg/redditlanguagemodeling/data/AmazonReviews/data",
                     data_path='amazon_v2.0/reviews.csv', split_file_name="my_user_split.csv",
                     embeddings_path="/data/ddmg/redditlanguagemodeling/data/AmazonReviews/data/amazon_v2.0/embeddings/my_user_split_clf_post_embeddings/",
                     train_split_num=0, val_split_num=1, test_split_num=2):
    data_df = pd.read_csv(os.path.join(data_dir, data_path),
                          dtype={'reviewerID': str, 'asin': str, 'reviewTime': str, 'unixReviewTime': int,
                                 'reviewText': str, 'summary': str, 'verified': bool, 'category': str,
                                 'reviewYear': int},
                          keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONNUMERIC)
    data_df = data_df.rename(columns={"reviewerID": "user", "overall": "review_score"})
    split_df = pd.read_csv(os.path.join(data_dir, 'amazon_v2.0', 'splits', split_file_name))
    data_df["split"] = split_df["split"]
    data_df = data_df[data_df["split"] != -1]
    train_df = data_df[data_df["split"] == train_split_num]
    val_df = data_df[data_df["split"] == val_split_num]
    test_df = data_df[data_df["split"] == test_split_num]
    # load embeddings
    split_name = split_file_name.split('.')[0]
    train_embeds = np.load(os.path.join(embeddings_path, "train_{}_split".format(split_name), "embeddings.npy"),
                           allow_pickle=True)
    train_df["text_embedding"] = list(train_embeds)
    val_embeds = np.load(os.path.join(embeddings_path, "val_{}_split".format(split_name), "embeddings.npy"),
                         allow_pickle=True)
    val_df["text_embedding"] = list(val_embeds)
    test_embeds = np.load(os.path.join(embeddings_path, "test_{}_split".format(split_name), "embeddings.npy"),
                          allow_pickle=True)
    test_df["text_embedding"] = list(test_embeds)
    return data_df, train_df, val_df, test_df


def filter_select_people(data_df, target_people_file="/data/ddmg/redditlanguagemodeling/results/amazon_reviews/clf/from_embeds/person_specific_my_split_n_500/selected_people.txt"):
    people = get_select_people(target_people_file)
    data_df = data_df[data_df["user"].isin(people)]
    return data_df


def get_select_people(target_people_file="/data/ddmg/redditlanguagemodeling/results/amazon_reviews/clf/from_embeds/person_specific_my_split_n_500/selected_people.txt"):
    with open(target_people_file, 'r') as f:
        people = f.read().splitlines()
    return people


def get_user_info(data_df):
    review_counts_by_user = data_df[["user", "review_score"]].groupby(["user"]).count()
    score_dist_by_user = get_review_counts(data_df)


def get_review_counts(data_df):
    score_df = data_df[["user", "review_score"]]
    count_fns = []
    for score in range(1, 6):
        count_fns.append(CountScore(score))
    score_dist_by_user = score_df.groupby(["user"]).agg(count_fns)
    return score_dist_by_user


def get_review_distr(data_df):
    score_dist_by_user = get_review_counts(data_df)
    #score_dist_by_user["total_score"] = score_dist_by_user.apply(lambda x: )


class CountScore:
    def __init__(self, score):
        self.score = score

    def __call__(self, x):
        return sum(x == self.score)
