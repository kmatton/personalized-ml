"""
Utility functions to support running experiments.
"""
import json
import math
import os
import sys
from shutil import copyfile

import pandas as pd
from transformers import HfArgumentParser, TrainingArguments
from IPython import embed

from argument_parsing.model_args import ModelArguments
from argument_parsing.data_args import DataArguments
from argument_parsing.experiment_args import ExperimentArguments
from argument_parsing.multi_run_experiment_args import MultiRunArguments


def parse_args():
    parser = HfArgumentParser((MultiRunArguments, ExperimentArguments, ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our argument_parsing.
        mr_args, exp_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        mr_args, exp_args, model_args, data_args, training_args, = parser.parse_args_into_dataclasses()
    return mr_args, exp_args, model_args, data_args, training_args


def save_args(output_dir):
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        fname = os.path.basename(sys.argv[1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        copyfile(os.path.abspath(sys.argv[1]), os.path.join(output_dir, fname))
    else:
        with open(os.path.join(output_dir, "command_args.json"), 'w') as f:
            json.dump(sys.argv, f)


def loss_to_perplexity(loss):
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity
