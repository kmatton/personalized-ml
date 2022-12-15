"""
Pytorch modules for computing loss functions where samples have different weights.
"""
import torch.nn as nn
from IPython import embed


class SampleWeightedLoss(nn.Module):
    """
    Take weighted sum of sample metrics.
    """
    def __init__(self, base_loss_fn):
        """
        :param base_loss_fn: core loss function to use in computing loss per sample (e.g., CrossEntropyLoss)
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn

    def sample_loss_fn(self, logits, labels):
        loss = self.base_loss_fn(logits, labels)
        return loss

    def forward(self, logits, labels, weights):
        per_sample_loss = self.sample_loss_fn(logits, labels)
        # compute weighted loss across all samples
        if weights is not None:
            loss = (weights * per_sample_loss).sum()
        else:
            loss = per_sample_loss.mean()
        return loss


class SampleWeightedMLMLoss(SampleWeightedLoss):
    """
    Same as above but computes loss per sample as mean loss per target word in sample.
    """
    def __init__(self, base_loss_fn, vocab_size):
        super().__init__(base_loss_fn)
        self.vocab_size = vocab_size

    def sample_loss_fn(self, logits, labels):
        masked_lm_loss = self.base_loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        # get sum of metrics across tokens in each sample
        masked_lm_loss = masked_lm_loss.view(labels.shape).sum(dim=1)
        # get counts of non-ignored words in each sample
        non_ignore_counts = (labels != -100).sum(dim=1)
        non_ignore_counts[non_ignore_counts == 0] = 1
        # divide metrics by counts of words that contributed to the loss
        masked_lm_loss /= non_ignore_counts
        return masked_lm_loss
