"""
Functions for adding fixed sample weights to data.
"""
import numpy as np
from IPython import embed


def get_fixed_sample_weights(dataset, sample_weight_scheme, sample_weight_target_col, sample_weight_target_val,
                             weight_vals):
    if sample_weight_scheme == "fixed_local_global":
        alpha = weight_vals[0]  # only one weight will be provided
        assert 0 <= alpha <= 1, "alpha needs to be [0,1]"
        is_target = np.array(dataset[sample_weight_target_col]) == sample_weight_target_val
        weights = np.zeros(len(dataset))
        weights[is_target] = alpha / sum(is_target)
        weights[~is_target] = (1 - alpha) / sum(~is_target)
        # just return weights because adding them as a column is slow
        return weights
    print("ERROR: unrecognized sample weight scheme {}".format(sample_weight_scheme))
    print("Exiting...")
    exit(1)


def add_fixed_sample_weights_old(dataset, sample_weight_scheme, sample_weight_target, batch_size):
    if sample_weight_scheme.split("_")[1] == "alpha":
        alpha = float(sample_weight_scheme.split("_")[2])
        assert 0 <= alpha <= 1, "alpha needs to be [0,1]"
        target_type = sample_weight_scheme.split("_")[0]
        is_target = np.array(dataset[target_type]) == sample_weight_target
        weights = np.zeros(len(dataset))
        # scale weights so that sample with max weight is 1/batch size
        max_weight = 1.0 / (batch_size)
        if alpha >= 1 - alpha:
            weights[is_target] = max_weight
            other_w = max_weight * (1 - alpha) / alpha
            weights[~is_target] = other_w
            print("weighting {} data {} and other data {}".format(sample_weight_target, max_weight, other_w))
        else:
            weights[~is_target] = max_weight
            other_w = max_weight * alpha / (1 - alpha)
            weights[is_target] = other_w
            print("weighting {} data {} and other data {}".format(sample_weight_target, other_w, max_weight))
        # NOTE: this is hacky, may want to fix
        # add dummy weights if not using full dataset because dataset internal memory table thinks it still
        # has the original length
        if len(dataset.data['id']) != len(weights):
            diff = len(dataset.data['id']) - len(weights)
            weights = list(weights) + [0] * diff
        dataset = dataset.add_column(name='weight', column=list(weights))
        return dataset
    print("ERROR: unrecognized sample weight scheme {}".format(sample_weight_scheme))
    print("Exiting...")
    exit(1)