"""
Class for data argument_parsing
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset: str = field(
        default="amazon_reviews",
        metadata={
            "help": "dataset to use"
        }
    )
    data_dir: str = field(
        default="/data/ddmg/redditlanguagemodeling/data/AmazonReviews/data/amazon_v2.0/",
        metadata={
            "help": "directory where data is"
        }
    )
    raw_data_file: str = field(
        default="reviews.csv",
        metadata={"help": "Name of CSV file with raw dataset."}
    )
    processed_data_dir: str = field(
        default="amazon_reviews_processed",
        metadata={"help": "name of directory where processed dataset is stored"}
    )
    split_file: Optional[str] = field(
        default=None,
        metadata={"help": "Name of CSV specifying data to include and train/test/val split."}
    )
    user_col: str = field(
        default="user",
        metadata={"help": "column name for column with user IDs (used for personalization experiments)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    personalize_domain_name: str = field(
        default='user',
        metadata={
            "help": "name of column that contains variable used to for identifying sub-populations to personalize based on"
        }
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        }
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        }
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker experiments, truncate the number of prediction examples to this "
            "value if set."
        }
    )
    select_random_train: bool = field(
        default=False,
        metadata={"help": "If True, select random samples instead of first samples when limiting # of train samples."}
    )
    select_random_val: bool = field(
        default=False,
        metadata={"help": "If True, select random samples instead of first samples when limiting # of val samples."}
    )
    select_random_test: bool = field(
        default=False,
        metadata={"help": "If True, select random samples instead of first samples when limiting # of test samples."}
    )
    save_dataset: bool = field(
        default=True,
        metadata={
            "help": "If True, will save processed dataset so can be used in future experiments."
        }
    )
    filter_train_by: Optional[str] = field(
        default=None,
        metadata={
            "help": "If provided, filter train dataset by this attributes."
        }
    )
    train_keep_vals: str = field(
        default_factory=lambda: [],
        metadata={"help": "list of values to keep when filtering"}
    )
    filter_val_by: Optional[str] = field(
        default=None,
        metadata={
            "help": "If provided, filter val dataset by this attributes."
        }
    )
    val_keep_vals: str = field(
        default_factory=lambda: [],
        metadata={"help": "list of values to keep when filtering"}
    )
    filter_test_by: Optional[str] = field(
        default=None,
        metadata={
            "help": "If provided, filter test dataset by this attributes."
        }
    )
    test_keep_vals: str = field(
        default_factory=lambda: [],
        metadata={"help": "list of values to keep when filtering"}
    )
    report_columns_discrete: Optional[str] = field(
        default_factory=lambda: [],
        metadata={"help": "discrete columns in dataset to report statistics on"}
    )
    report_columns_cont: Optional[str] = field(
        default_factory=lambda: [],
        metadata={"help": "continuous columns in dataset to report statistics on"}
    )
    ##### ARGUMENTS FOR LANGUAGE DATA #####
    tokenizer_name: Optional[str] = field(
        default="distilbert-base-uncased",
        metadata={"help": "Pretrained tokenizer name or path. Only relevant when using language data."}
    )
    tokenizer_cache_dir: Optional[str] = field(
        default="/data/ddmg/redditlanguagemodeling/cached/distilbert_clf",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    ### ARGUMENTS FOR MLM ####
    fix_test_labels: bool = field(
        default=True,
        metadata={"help": "Whether or not to use pred/test data with fixed labels."}
    )
    fix_val_labels: bool = field(
        default=False,
        metadata={"help": "Whether or not to use eval data with fixed labels."}
    )
    fix_train_labels: bool = field(
        default=False,
        metadata={"help": "Whether or not to use train data with fixed labels."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    ### ARGUMENTS RELATED TO USER EMBEDDINGS ###
    add_user_embeds: bool = field(
        default=False,
        metadata={"help": "If true, add user embeddings to dataset."}
    )
    user_embed_method: Optional[str] = field(
        default=None,
        metadata={"help": "method to use to use to generate user embedding initializations."}
    )
