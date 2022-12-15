"""
Script for launching multiple experiments.
"""
import os
import time

from IPython import embed

from common.factories import get_dataset, get_embed_model, get_cluster_model
from runners.run_exp import ExpRunner
from runners.runner_utils import parse_args, save_args


def main():
    mr_args, exp_args, model_args, data_args, training_args = parse_args()
    save_args(mr_args.base_output_dir)
    if mr_args.personalize_targets is not None:
        with open(mr_args.personalize_targets, 'r') as f:
            targets = f.read().splitlines()
    else:  # otherwise get from data
        dataset = get_dataset(data_args)
        # embed data
        if exp_args.embed_data:
            embed_model = get_embed_model(exp_args)
            dataset.embed_data(embed_model, exp_args.embed_path)
        # cluster data
        if exp_args.cluster_data:
            cluster_model = get_cluster_model(exp_args)
            dataset.cluster_data(cluster_model, exp_args.cluster_path, exp_args.cluster_train_data)
        targets = set(dataset.test_data[exp_args.personalize_by])
    for idx, target in enumerate(targets):
        init_time = time.time()
        print('Starting experiment with {} as target: {}/{}'.format(target, idx + 1, len(targets)))
        training_args.output_dir = os.path.join(mr_args.base_output_dir, str(target))
        exp_args.personalize_target = target
        exp_runner = ExpRunner(exp_args, model_args, data_args, training_args)
        exp_runner.run()
        print('Finished experiment for target in {} in time {}'.format(target, time.time() - init_time))


if __name__ == "__main__":
    main()