"""
Class for model argument_parsing.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model: str = field(
        default="sequence_classification",
        metadata={
            "help": "model class to use"
        },
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    config_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="/data/ddmg/redditlanguagemodeling/cached/distilbert_clf",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    n_freeze_layers: Optional[int] = field(
        default=0,
        metadata={"help": "number of layers to freeze if using pretrained model"}
    )
    #### Arguments for Basic NN ####
    hidden_layer_sizes: str = field(
        default_factory=lambda: [],
        metadata={"help": "List of sizes of hidden layers to include. If None provided, will not use hidden layers."}
    )
    ### Arguments for User Embed Model
    user_embed_model: str = field(
        default="user_embed_nn",
        metadata={
            "help": "model class to use"
        },
    )
    user_embed_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    user_embed_hidden_layer_sizes: str = field(
        default_factory=lambda: [],
        metadata={"help": "List of sizes of hidden layers to include. If None provided, will not use hidden layers."}
    )
    user_embed_dim: str = field(
        default=None,
        metadata={"help": "dimension of learned user embeddings"}
    )
    data_parallel: bool = field(
        default=True,
        metadata={
            "help": "whether to wrap model with data parallel or not"
        }
    )