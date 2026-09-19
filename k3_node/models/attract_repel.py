import keras
from keras import ops


class ARLinkPredictor(keras.layers.Layer):
    r"""Link predictor using Attract-Repel embeddings from the paper
    `"Pseudo-Euclidean Attract-Repel Embeddings for Undirected Graphs"
    <https://arxiv.org/abs/2106.09671>`_.

    This model splits node embeddings into: attract and repel.
    The edge prediction score is computed as the dot product of attract
    components minus the dot product of repel components.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of hidden embeddings.
        out_channels (int, optional): Size of output embeddings. If set to
            `None`, will default to `hidden_channels`. (default: `None`)
        num_layers (int): Number of message passing layers. (default: `2`)
        dropout (float): Dropout probability. (default: `0.0`)
        attract_ratio (float): Ratio to use for attract component. Must be
            between 0 and 1. (default: `0.5`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels=None,
                num_layers=2, dropout=0.0, attract_ratio=0.5, **kwargs):
        super().__init__(**kwargs)

        if out_channels is None:
            out_channels = hidden_channels

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout

        if not 0 <= attract_ratio <= 1:
            raise ValueError(f"attract_ratio must be between 0 and 1, got {attract_ratio}")

        self.attract_ratio = attract_ratio
        self.attract_dim = int(out_channels * attract_ratio)
        self.repel_dim = out_channels - self.attract_dim

        self.lins = [keras.layers.Dense(hidden_channels)]
        for _ in range(num_layers - 2):
            self.lins.append(keras.layers.Dense(hidden_channels))

        self.lin_attract = keras.layers.Dense(self.attract_dim)
        self.lin_repel = keras.layers.Dense(self.repel_dim)
        self.dropout = keras.layers.Dropout(dropout) if dropout > 0.0 else None

        self.lins[0].build((None, in_channels))
        for lin in self.lins[1:]:
            lin.build((None, hidden_channels))
        self.lin_attract.build((None, hidden_channels))
        self.lin_repel.build((None, hidden_channels))

    def encode(self, x, *args, **kwargs):
        r"""Encode node features into attract-repel embeddings."""
        for lin in self.lins:
            x = lin(x)
            x = ops.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)

        attract_x = self.lin_attract(x)
        repel_x = self.lin_repel(x)

        return attract_x, repel_x

    def decode(self, attract_z, repel_z, edge_index):
        r"""Decode edge scores from attract-repel embeddings."""
        row, col = edge_index[0], edge_index[1]
        attract_z_row = ops.take(attract_z, row, axis=0)
        attract_z_col = ops.take(attract_z, col, axis=0)
        repel_z_row = ops.take(repel_z, row, axis=0)
        repel_z_col = ops.take(repel_z, col, axis=0)

        attract_score = ops.sum(attract_z_row * attract_z_col, axis=1)
        repel_score = ops.sum(repel_z_row * repel_z_col, axis=1)

        return attract_score - repel_score

    def call(self, x, edge_index):
        attract_z, repel_z = self.encode(x)
        return ops.sigmoid(self.decode(attract_z, repel_z, edge_index))

    def calculate_r_fraction(self, attract_z, repel_z):
        r"""Calculate the R-fraction (proportion of energy in repel space)."""
        attract_norm_squared = ops.sum(ops.square(attract_z))
        repel_norm_squared = ops.sum(ops.square(repel_z))

        r_fraction = repel_norm_squared / (attract_norm_squared + repel_norm_squared + 1e-10)

        return float(ops.convert_to_numpy(r_fraction))
