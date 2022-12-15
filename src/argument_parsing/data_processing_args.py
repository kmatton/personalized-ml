"""
Class for data argument_parsing
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataProcessArguments:
    """
    Arguments pertaining to how to process a dataset.
    """
    cluster_data: bool = field(
        default=False,
        metadata={"help": "If true, cluster data and save cluster assignments."}
    )
    cluster_features: Optional[str] = field(
        default=None,
        metadata={"help": "attribute of data to cluster examples based on"}
    )
    num_clusters: int = field(
        default=5,
        metadata={"help": "Number of clusters to use"}
    )
    ##### ARGUMENTS FOR LANGUAGE DATA #####
    apply_embed_preprocess: bool = field(
        default=False,
        metadata={"help": "If true, use embed_model to generate embeddings of raw input data."}
    )
    embed_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model to use to generate input data embeddings."}
    )
    embed_model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of model at embed_model_path. (e.g., masked_lm, sequence_classification)"}
    )
