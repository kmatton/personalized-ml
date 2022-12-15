"""
Script for launching multiple experiments with different params.
"""
import os
import time

from IPython import embed

from runners.run_exp import ExpRunner
from runners.runner_utils import parse_args, save_args


PARAM_DICT = {
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "meta_lr": [1, 1e-1, 1e-2, 1e-3],
}


def main():
    num_exps = len(PARAM_DICT["learning_rate"]) * len(PARAM_DICT["meta_lr"])
    mr_args, exp_args, model_args, data_args, training_args = parse_args()
    base_dir = training_args.output_dir
    exp_num = 1
    for lr in PARAM_DICT["learning_rate"]:
        training_args.learning_rate = lr
        for meta_lr in PARAM_DICT["meta_lr"]:
            exp_args.meta_lr = meta_lr
            init_time = time.time()
            print("Starting experiment with pred LR {} and meta LR {}".format(lr, meta_lr))
            training_args.output_dir = os.path.join(base_dir, "predLR_{}_metaLR_{}".format(lr, meta_lr))
            exp_runner = ExpRunner(exp_args, model_args, data_args, training_args)
            exp_runner.run()
            print("Finished experiment {}/{} in time {}".format(exp_num, num_exps, time.time() - init_time))
            exp_num += 1


if __name__ == "__main__":
    main()
