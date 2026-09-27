from typing import Optional
import numpy as np
import keras
from keras import ops

from k3_node.layers.aggr.base import from_dense_batch, to_dense_batch

from k3_node.layers.conv import GCNConv
from k3_node.layers.attention import SGFormerAttention


class GraphModule(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.convs = []
        self.fcs = [keras.layers.Dense(hidden_channels)]
        self.bns = [keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)]

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))

        self.drop = keras.layers.Dropout(dropout)
        self.dropout_rate = dropout

    def call(self, x, edge_index, training=False):
        x = self.fcs[0](x)
        x = self.bns[0](x, training=training)
        x = ops.relu(x)
        x = self.drop(x, training=training)
        last_x = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i + 1](x, training=training)
            x = ops.relu(x)
            x = self.drop(x, training=training)
            x = x + last_x
        return x


class SGModule(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attns = []
        self.fcs = [keras.layers.Dense(hidden_channels)]
        self.bns = [keras.layers.LayerNormalization()]

        for _ in range(num_layers):
            self.attns.append(
                SGFormerAttention(hidden_channels, num_heads, hidden_channels)
            )
            self.bns.append(keras.layers.LayerNormalization())

        self.drop = keras.layers.Dropout(dropout)
        self.dropout_rate = dropout

    def call(self, x, batch=None, training=False):
        if batch is None:
            batch = ops.zeros((ops.shape(x)[0],), dtype="int64")

        # Group the nodes by graph (static shapes: works under jit)
        batch = ops.cast(batch, "int32")
        indices = ops.argsort(batch)
        rev_perm = ops.argsort(indices)
        batch_sorted = ops.take(batch, indices, axis=0)
        x_dense, mask = to_dense_batch(ops.take(x, indices, axis=0), batch_sorted)

        layer_ = []

        # input MLP layer
        x = self.fcs[0](x_dense)
        x = self.bns[0](x, training=training)
        x = ops.relu(x)
        x = self.drop(x, training=training)

        layer_.append(x)

        for i, attn in enumerate(self.attns):
            x = attn(x, mask)
            x = (x + layer_[i]) / 2.0
            x = self.bns[i + 1](x, training=training)
            x = ops.relu(x)
            x = self.drop(x, training=training)
            layer_.append(x)

        unsorted_x_mask = ops.take(from_dense_batch(x, batch_sorted), rev_perm, axis=0)
        return unsorted_x_mask


class SGFormer(keras.Model):
    r"""The sgformer module from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        in_channels (int): Input channels.
        hidden_channels (int): Hidden channels.
        out_channels (int): Output channels.
        trans_num_layers (int, optional): The number of layers for all-pair attention.
            (default: :obj:`2`)
        trans_num_heads (int, optional): The number of heads for attention.
            (default: :obj:`1`)
        trans_dropout (float, optional): Global dropout rate.
            (default: :obj:`0.5`)
        gnn_num_layers (int, optional): The number of layers for GNN.
            (default: :obj:`3`)
        gnn_dropout (float, optional): GNN dropout rate.
            (default: :obj:`0.5`)
        graph_weight (float, optional): The weight balance global and gnn module.
            (default: :obj:`0.5`)
        aggregate (str, optional): Aggregate type (:obj:`'add'` or :obj:`'cat'`).
            (default: :obj:`'add'`)

    Example:
        ```python
        import numpy as np
        from k3_node.models import SGFormer

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges

        model = SGFormer(in_channels=8, hidden_channels=32, out_channels=4, trans_num_layers=1,
                         trans_num_heads=1, gnn_num_layers=1)
        print(tuple(model(x, edge_index).shape))  # (10, 4)
        ```
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        trans_num_layers: int = 2,
        trans_num_heads: int = 1,
        trans_dropout: float = 0.5,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.5,
        graph_weight: float = 0.5,
        aggregate: str = "add",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.graph_weight = graph_weight
        self.aggregate = aggregate

        self.trans_conv = SGModule(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = GraphModule(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
        )

        if aggregate == "add":
            self.fc = keras.layers.Dense(out_channels)
        elif aggregate == "cat":
            self.fc = keras.layers.Dense(out_channels)
        else:
            raise ValueError(f"Invalid aggregate type: {aggregate}")

    def call(self, x, edge_index, batch: Optional[any] = None, training=False):
        x1 = self.trans_conv(x, batch, training=training)
        x2 = self.graph_conv(x, edge_index, training=training)
        if self.aggregate == "add":
            x = self.graph_weight * x2 + (1.0 - self.graph_weight) * x1
        else:
            x = ops.concatenate([x1, x2], axis=1)
        x = self.fc(x)
        return ops.log_softmax(x, axis=-1)

