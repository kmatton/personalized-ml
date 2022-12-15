"""
Trainer state class to keep track of training progress.
"""
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from IPython import embed


@dataclass
class TrainerState:
    # the value of the best metric encountered so far
    best_metric: Optional[float] = None
    global_step: int = 0
    log_history: List[Dict[str, float]] = None
    total_train_loss: float = 0

    def maybe_update_best_metric(self, metrics, greater_is_better, metric_to_check):
        updated = False
        # if current metric is better than best metric, update it
        operator = np.greater if greater_is_better else np.less
        metric_value = metrics.get(metric_to_check)
        if self.best_metric is None or (operator(metric_value, self.best_metric)):
            self.best_metric = metric_value
            updated = True
        return updated

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path):
        """Save the content of this instance in JSON format inside :obj:`json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def reset(self):
        self.global_step = 0
        self.best_metric = None
        self.log_history = []
        self.total_train_loss = 0
