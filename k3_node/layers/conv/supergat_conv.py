import math
import keras
from keras import ops
from keras.layers import Dense, Dropout

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import (
    add_self_loops,
    extend_mask_for_self_loops,
    mask_edge_logits,
    remove_self_loops_masked,
    softmax,
)
from k3_node.ops.creation import full


class SuperGATConv(MessagePassing):
    r"""The self-supervised graph attentional operator from the
    `"How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision"
    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper.

    Args:
        attention_type (str): ``"MX"`` (mixed GO/DP) or ``"SD"`` (scaled dot-product).
        neg_sample_ratio (float): Negative (random) pairs per positive edge in the attention loss.
        edge_sample_ratio (float): Fraction of edges used as positives in the attention loss.
        attention_loss_weight (float): If positive, the self-supervised attention loss is added to
            the model loss (``model.losses``) with this weight while training, so ``fit`` optimizes
            it automatically. PyG's example uses ``4.0``. (default: ``0.0``)

    Example:
        ```python
        import numpy as np
        from k3_node.layers import SuperGATConv

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges

        layer = SuperGATConv(in_channels=8, out_channels=16, heads=2)
        out = layer(x, edge_index)
        print(tuple(out.shape))  # (10, 32)
        ```
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        attention_type: str = "MX",
        neg_sample_ratio: float = 0.5,
        edge_sample_ratio: float = 1.0,
        is_undirected: bool = False,
        attention_loss_weight: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        assert attention_type in ["MX", "SD"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.is_undirected = is_undirected
        self.use_bias = bias

        self.attention_loss_weight = attention_loss_weight
        self.lin = Dense(heads * out_channels, use_bias=False)
        self.dropout = Dropout(dropout)
        self.seed_generator = keras.random.SeedGenerator()
        self._last_attention_loss = None

        if self.attention_type == "MX":
            self.att_l = self.add_weight(
                shape=(1, heads, out_channels),
                initializer="glorot_uniform",
                name="att_l",
            )
            self.att_r = self.add_weight(
                shape=(1, heads, out_channels),
                initializer="glorot_uniform",
                name="att_r",
            )
        else:
            self.att_l = None
            self.att_r = None

        if bias:
            out_dim = heads * out_channels if concat else out_channels
            self.bias = self.add_weight(
                shape=(out_dim,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.lin.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, edge_index=None, training=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                x, edge_index = inputs
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        num_nodes = ops.shape(x)[0]
        keep_mask = None
        if self.add_self_loops:
            edge_index, _, keep_mask = remove_self_loops_masked(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            keep_mask = extend_mask_for_self_loops(keep_mask, num_nodes)

        x = self.lin(x)
        x = ops.reshape(x, (-1, self.heads, self.out_channels))

        out = self.propagate(edge_index, x=x, keep_mask=keep_mask, training=training, size=(num_nodes, num_nodes))

        if training:
            loss = self._attention_loss(x, edge_index, num_nodes)
            if self.attention_loss_weight:
                self.add_loss(self.attention_loss_weight * loss)
            from k3_node.layers.conv.utils import is_tracing
            self._last_attention_loss = None if is_tracing(loss) else loss

        if self.concat:
            out = ops.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = ops.mean(out, axis=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i, x_j, index=None, size_i=None, keep_mask=None, training=None):
        if self.attention_type == "MX":
            logits = ops.sum(x_i * x_j, axis=-1)
            alpha = ops.sum(x_j * self.att_l, axis=-1) + ops.sum(x_i * self.att_r, axis=-1)
            alpha = alpha * ops.sigmoid(logits)
        else:  # SD
            alpha = ops.sum(x_i * x_j, axis=-1) / math.sqrt(self.out_channels)

        alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = mask_edge_logits(alpha, keep_mask)
        alpha = softmax(alpha, index, num_nodes=size_i, dim=0)
        alpha = self.dropout(alpha, training=training)
        return x_j * ops.expand_dims(alpha, -1)

    def _attention_logits(self, x_i, x_j):
        logits = ops.sum(x_i * x_j, axis=-1)
        if self.attention_type == "SD":
            logits = logits / math.sqrt(self.out_channels)
        return logits

    def _attention_loss(self, x, edge_index, num_nodes):
        # Self-supervised attention loss: attention logits should separate real edges (label 1)
        # from random node pairs (label 0). Edges are kept with probability `edge_sample_ratio` and
        # random pairs with probability `neg_sample_ratio * edge_sample_ratio` (the expected counts
        # used by PyG); weighting instead of slicing keeps all shapes static for XLA / jax.jit.
        edge_index = ops.cast(edge_index, "int32")
        num_edges = ops.shape(edge_index)[1]
        neg = keras.random.randint(ops.shape(edge_index), 0, num_nodes, seed=self.seed_generator, dtype="int32")
        pairs = ops.concatenate([edge_index, neg], axis=1)
        logits = ops.mean(self._attention_logits(ops.take(x, pairs[1], axis=0), ops.take(x, pairs[0], axis=0)), axis=-1)
        labels = ops.concatenate([ops.ones((num_edges,)), ops.zeros((num_edges,))])
        keep = keras.random.uniform(ops.shape(labels), seed=self.seed_generator) < ops.concatenate([
            full((num_edges,), self.edge_sample_ratio),
            full((num_edges,), self.neg_sample_ratio * self.edge_sample_ratio),
        ])
        weights = ops.cast(keep, "float32")
        losses = ops.binary_crossentropy(labels, logits, from_logits=True)
        return ops.sum(losses * weights) / ops.maximum(ops.sum(weights), 1.0)

    def get_attention_loss(self):
        r"""The self-supervised attention loss of the last training call (as in PyG)."""
        return self._last_attention_loss
