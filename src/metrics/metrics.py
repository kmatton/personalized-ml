"""
Class for computing performance metrics.
"""
import numpy as np
from IPython import embed


class ComputeMetrics:
    def __init__(self, metric_names):
        self.metric_names = metric_names

    def __call__(self, eval_prediction):
        preds = eval_prediction.predictions
        labels = eval_prediction.label_ids
        preds = preds[0] if isinstance(preds, tuple) else preds
        if preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
        else:  # binary prediction
            preds = preds > 0.5
        result_dict = {}
        if "accuracy" in self.metric_names:
            result_dict["accuracy"] = (preds == labels).astype(np.float32).mean().item()
        if "per_class_accuracy" in self.metric_names:
            if len(preds.shape) == 1 and len(labels.shape) > 1:
                preds = preds[:, None]
            classes = sorted(np.unique(np.concatenate([preds, labels])))
            for _class in classes:
                is_class = (labels == _class)
                class_preds = preds[is_class]
                result_dict["{}_accuracy".format(_class)] = (class_preds == _class).astype(np.float32).mean().item()
        return result_dict
