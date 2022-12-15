"""
Intended for Training Main Prediction Model.

Class that inherits from HF's trainer class, but with some extensions:

(1) Enables more efficient evaluation of model predictions/accuracy.
* Ran into memory issues when using compute_metrics to compute accuracy because HF implementation stores all
  logits, concatenates them, and then computes accuracy.
* In my new trainer, I've changed it so that only the indices of the max logit are stored rather than all logits.

(2) Enables output of per-sample loss.

(3) Includes an optional deterministic mode where training sampler isn't random.

(4) Includes option to compute 'even weight' loss - average loss per sample rather than per word.

"""
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.nn import CrossEntropyLoss

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.dependency_versions_check import dep_version_check
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.utils.modeling_auto_mapping import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
from IPython import embed

from transformers import Trainer


_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
else:
    import torch.distributed as dist

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


class MyTrainer(Trainer):
    def __init__(self, loss_fn, deterministic_mode, use_weighted_loss, use_weighted_sampler, train_sample_weights=None,
                 **kwargs):
        self.loss_fn = loss_fn
        self.deterministic_mode = deterministic_mode
        self.use_weighted_loss = use_weighted_loss
        self.use_weighted_sampler = use_weighted_sampler
        self.train_sample_weights = train_sample_weights
        super().__init__(**kwargs)

    def _remove_unused_columns(self, dataset: datasets.Dataset, description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep the arguments it accepts
            signature1 = inspect.signature(self.model.forward)
            # Inspect loss function signature; also keep these arguments
            # note: need to do .module because loss_fn is wrapped in data parallel
            signature2 = inspect.signature(self.loss_fn.module.forward)
            self._signature_columns = list(set(signature1.parameters.keys()).union(set(signature2.parameters.keys())))
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overriding compute loss function to use custom loss function
        """
        if self.label_smoother is not None:
            print("ERROR: Custom Trainer is currently not implemented to work with label_smoother.")
            print("Exiting...")
            exit(1)
        labels = inputs.pop("labels")
        outputs = model(**inputs)  # outputs logits and loss per batch on each GPU
        # compute loss
        logits = outputs['logits']
        loss_inputs = dict(
            logits=logits,
            labels=labels
        )
        loss = self.loss_fn(**loss_inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override to allow for (1) deterministic mode to use sequential sampler and (2) weighted sampler
        """
        if self.deterministic_mode:
            print('using sequential sampler for train - deterministic mode')
            return SequentialSampler(self.train_dataset)
        elif self.use_weighted_sampler:
            # sample (# of dataset) examples so that same amount of training data is used as in other experiments
            return WeightedRandomSampler(self.train_sample_weights, len(self.train_dataset))
        else:
            return RandomSampler(self.train_dataset)
