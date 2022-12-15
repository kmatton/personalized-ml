"""
Abstract base class for model that computes sample weights.
"""
import torch
import torch.nn as nn


class SampleWeightModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        raise NotImplementedError


class EvenSampleWeightModel(SampleWeightModel):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        weight = 1 / inputs.shape[0]
        return torch.full(inputs.shape[0], weight)


class FixedSampleWeightModel(SampleWeightModel):
    def __init__(self, id2weight):
        self.id2weight = id2weight
        super().__init__()

    def forward(self, inputs):
        ids = inputs["id"]
        weights = [self.id2weight[sample_id] for sample_id in ids]
        return torch.tensor(weights)
