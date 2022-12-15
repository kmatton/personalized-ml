"""
Network to learn user embeddings.
"""
import torch.nn as nn

from models.basic_nn import BasicNN


class UserEmbeddingNN(BasicNN):
    """
    Model for learning user embeddings.
    """
    def __init__(self, input_dim, n_outputs, hidden_layers, dropout_prob=0.2, initializer_range=0.02):
        super().__init__(input_dim, n_outputs, hidden_layers, dropout_prob, initializer_range)
        self.batch_norm = nn.BatchNorm1d(n_outputs)

    def forward(self, x):
        x = self.layers(x)
        return self.batch_norm(x)
