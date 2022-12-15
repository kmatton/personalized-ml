"""
Network to learn embeddings.
"""
import torch.nn as nn

from models.basic_nn import BasicNN


class EmbeddingNN(nn.Module):
    """
    Notes: not sure that we want the layer norm part
    - also should I just do one-hot vectors as input rather than this embedding layer?
    - maybe this unnecessary because the inputs are all single embeddings, not a list of embeddings
    """
    def __init__(self, dictionary_size, init_embedding_dim, out_embedding_dim, hidden_layers, dropout_prob=0.2,
                 initializer_range=0.02):
        super().__init__()
        self.dictionary_size = dictionary_size
        self.init_embedding_dim = init_embedding_dim
        self.out_embedding_dim = out_embedding_dim
        # initialize network to map from input IDS to initialize
        self.embeddings = nn.Embedding(dictionary_size, init_embedding_dim)
        # initialize network to learn transformed embeddings
        self.transform_embed_net = None
        if len(hidden_layers):
            self.transform_embed_net = BasicNN(input_dim=init_embedding_dim, n_outputs=out_embedding_dim,
                                               hidden_layers=hidden_layers, dropout_prob=dropout_prob,
                                               initializer_range=initializer_range)
        self.layer_norm = nn.LayerNorm(init_embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def init_weights(self):
        if self.transform_embed_net is not None:
            self.transform_embed_net.init_weights()

    def forward(self, input_ids):
