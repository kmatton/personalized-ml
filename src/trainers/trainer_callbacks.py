"""
Callbacks to use with the Trainer classes to customize the training loop.

Adapted from transformers library.
"""

import numpy as np


class EarlyStopping:
    def __init__(self, patience, threshold):
        self.patience = patience
        self.threshold = threshold
        self.patience_counter = 0

    def check_metric_value(self, metrics, best_metric, greater_is_better, metric_to_check):
        stop_training = False
        operator = np.greater if greater_is_better else np.less
        metric_value = metrics.get(metric_to_check)
        if best_metric is None or (
                operator(metric_value, best_metric)
                and abs(metric_value - best_metric) > self.threshold
        ):
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.patience:
            stop_training = True
        return stop_training

    def reset(self):
        self.patience_counter = 0