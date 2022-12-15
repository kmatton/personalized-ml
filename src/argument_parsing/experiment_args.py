"""
Class for general experiment argument_parsing.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentArguments:
    """
    Arguments pertaining to general experiment setup.
    """
    exp_type: str = field(
        default="generic",
        metadata={"help": "Type of experiment to run (generic, personalized, pretrained_models)"}
    )
    trainer: str = field(
        default="HFTrainer",
        metadata={"help": "Class of trainer to use."}
    )
    should_log: bool = field(
        default=False,
        metadata={"help": "Whether to output details of experiment."}
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print details of experiment."}
    )
    early_stopping: bool = field(
        default=False,
        metadata={"help": "If true, apply early stopping."}
    )
    num_runs: int = field(
        default=1,
        metadata={"help": "Number of times to run model."}
    )
    start_seed: int = field(
        default=42,
        metadata={"help": "Seed to use for first experiment run. Each of the next runs will increment this."}
    )
    deterministic_mode: bool = field(
        default=False,
        metadata={"help": "Whether to run the experiment in deterministic mode "
                          "(don't use random sampler for train data, etc.)"}
    )
    loss_fn: str = field(
        default="cross_entropy",
        metadata={"help": "Type of loss function to use."}
    )
    eval_by_group: Optional[str] = field(
        default=None,
        metadata={"help": "name of group to do group-wise evaluation by"}
    )
    group_vals: Optional[str] = field(
        default=None,
        metadata={"help": "path to file with names of groups to performance group-wise eval on. If not provided and "
                          "eval_by_group is set, will evaluate for all unique group values."}
    )
    generate_predictions: bool = field(
        default=False,
        metadata={"help": "If true, generate predictions on test data."}
    )
    eval_metrics: str = field(
        default_factory=lambda: [],
        metadata={"help": "metrics to use when assessing validation and test performance. "
                          "If none provided, will only report loss."}
    )
    use_weighted_loss: bool = field(
        default=False,
        metadata={"help": "Use in conjunction with sample weighting scheme. "
                          "If true, will weight loss function according to sample weights (if weights are provided)."}
    )
    use_weighted_sampler: bool = field(
        default=False,
        metadata={"help": "Use in conjunction with sample weighting scheme. "
                          "If true, will select samples with prob based on sample weights (if weights are provided)."}
    )
    es_patience: int = field(
        default=3,
        metadata={"help": "Patience to use for early stopping"}
    )
    es_threshold: float = field(
        default=0.001,
        metadata={"help": "Threshold to use for early stopping."}
    )
    ##### Arguments to for personalized/ person-specific model training
    personalize_col: Optional[str] = field(
        default=None,
        metadata={"help": "Name of column to distinguish sub-populations based on (e.g., user, category)."}
    )
    optimizer_type: str = field(
        default="Adam",
        metadata={"help": "Type of optimizer to use. Options: Adam or AdamW"}
    )
    person_select_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to file with names of sub-populations to train person-specific models for."}
    )
    select_n_persons: Optional[int] = field(
        default=None,
        metadata={"help": "If provided, will select a random size n subset "
                          "of personalization target vals to personalize for."}
    )
    save_selected: Optional[str] = field(
        default=None,
        metadata={"help": "If provided, will save list of personalization targets selected to file at the given path."}
    )
    skip_completed_people: bool = field(
        default=False,
        metadata={"help": "If True, will skip already completed personalization targets "
                          "when training person-specific models."}
    )
    personalize_target: str = field(
        default=None,
        metadata={"help": "Name of person/sub-population to use as the target in a personalized experiment."}
    )
    personalize_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "Strategy to use for personalization. (e.g., train_local, sample_weight)"}
    )
    ##### Arguments to use with sample weighting trainer
    sample_weight_scheme: Optional[str] = field(
        default=None,
        metadata={"help": "If provided, will weight each training sample according to the weighting scheme given."}
    )
    sample_weight_vals: Optional[float] = field(
        default_factory=lambda: [],
        metadata={"help": "Sample weight values to use with given sample weight scheme."}
    )
    sample_weight_alphas: Optional[float] = field(
        default_factory=lambda: [],
        metadata={"help": "Local data weight values to sweep over when doing fixed sample weighting schemes."}
    )
    #### ARGUMENTS FOR CLUSTERING DATA #####
    cluster_data: bool = field(
        default=False,
        metadata={"help": "If true, cluster data and save cluster assignments."}
    )
    cluster_feature: Optional[str] = field(
        default=None,
        metadata={"help": "attribute of data to cluster examples based on"}
    )
    num_clusters: int = field(
        default=5,
        metadata={"help": "Number of clusters to use"}
    )
    cluster_model: str = field(
        default="kmeans",
        metadata={"help": "Type of model to use to cluster data"}
    )
    cluster_assignments_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to output cluster assignmetns to (or read them from if they exist)."}
    )
    cluster_train_data: str = field(
        default="all",
        metadata={"help": "which splits of the data (e.g., all, train) to use for training clustering model."}
    )
    cluster_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to output cluster assignments (or read them from if they exist)."}
    )
    #### ARGUMENTS FOR EMBEDDING DATA ####
    embed_data: bool = field(
        default=False,
        metadata={"help": "If true, load pre-computed embeddings "
                          "or use embed_model to generate embeddings of raw input data."}
    )
    embed_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model to use to generate input data embeddings."}
    )
    embed_model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of model at embed_model_path. (e.g., masked_lm, sequence_classification)"}
    )
    embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to output data embeddings to (or read them from if they exist)"}
    )
    embed_model_config: Optional[str] = field(
        default=None,
        metadata={"help": "path to configuration file associated with embedding model"}
    )
    save_data_with_embeds_dir: Optional[str] = field(
        default=None,
        metadata={"help": "if provided, will save dataset with embeddings to the specified path."}
    )
    #### ARGUMENTS FOR LOOPING THROUGH PRE-TRAINED MODELS ####
    model_levels: Optional[str] = field(
        default_factory=lambda: [],
        metadata={"help": "List of levels of subdirectories to search through for pretrained models."}
    )
    #### Arguments for user weight trainer experiments ####
    user_embed_batch_size: int = field(
        default=64,
        metadata={"help": "batch size to use when running user embedding model"}
    )
    outer_steps: int = field(
        default=1000,
        metadata={"help": "max number of outer iterations in user weight training process"}
    )
    inner_steps: int = field(
        default=100,
        metadata={"help": "number of steps to train each pred model for during each inner iteration"}
    )
    user_embed_eval_steps: int = field(
        default=100,
        metadata={"help": "evaluate user embed model every this many steps"}
    )
    user_batch_size: int = field(
        default=20,
        metadata={"help": "number of users to train models for during a single inner iteration."}
    )
    user_embed_early_stopping: bool = field(
        default=True,
        metadata={"help": "if true, apply early stopping when training user embedding model"}
    )
    meta_lr: float = field(
        default=1e-2,
        metadata={"help": "learning rate for meta learning (i.e., user embed model)"}
    )
