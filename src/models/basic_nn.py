"""
Base class for basic NN model.
"""
import torch.nn as nn


class BasicNN(nn.Module):
    def __init__(self, input_dim, n_outputs, hidden_layers, dropout_prob=0.2, initializer_range=0.02):
        super().__init__()
        self.input_dim = input_dim
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.initializer_range = initializer_range
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.config = {"dropout_prob": dropout_prob, "initializer_range": initializer_range,
                       "input_dim": input_dim, "n_outputs": n_outputs, "hidden_layers": hidden_layers}
        self.layers = None
        self.construct_network()
        self.init_weights()

    def construct_network(self):
        # construct network
        layers = []
        if self.hidden_layers:
            layers = self._add_hidden_layers(layers)
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.input_dim, self.n_outputs)
            )

    def _add_hidden_layers(self, layers):
        layers.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_prob))
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_prob))
        layers.append(nn.Linear(self.hidden_layers[-1], self.n_outputs))
        return layers

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, embeddings):
        logits = self.layers(embeddings)
        outputs = {'logits': logits}
        return outputs
