"""
Script for generating text embeddings with pre-trained Language Model.
"""
import os
import sys
from transformers import HfArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
assert torch.cuda.is_available(), "selected gpus, but cuda not available"

from argument_parsing.data_args import DataArguments
from argument_parsing.model_args import ModelArguments
from common.factories import get_dataset, get_model


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    dataset = get_dataset(data_args)
    model = get_model(model_args, dataset.d_out)
    dataset.embed_data(model, save=True)


if __name__ == "__main__":
    main()