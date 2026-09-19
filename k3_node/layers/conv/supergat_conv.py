import math
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import remove_self_loops, add_self_loops, softmax


class SuperGATConv(MessagePassing):
    r"""The self-supervised graph attentional operator from the
    `"How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision"
    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper.
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

        self.lin = Dense(heads * out_channels, use_bias=False)

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

    def call(self, inputs, edge_index=None, **kwargs):
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
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        x = self.lin(x)
        x = ops.reshape(x, (-1, self.heads, self.out_channels))

        out = self.propagate(edge_index, x=x, size=(num_nodes, num_nodes))

        if self.concat:
            out = ops.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = ops.mean(out, axis=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i, x_j, index=None, size_i=None):
        if self.attention_type == "MX":
            logits = ops.sum(x_i * x_j, axis=-1)
            alpha = ops.sum(x_j * self.att_l, axis=-1) + ops.sum(x_i * self.att_r, axis=-1)
            alpha = alpha * ops.sigmoid(logits)
        else:  # SD
            alpha = ops.sum(x_i * x_j, axis=-1) / math.sqrt(self.out_channels)

        alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index, num_nodes=size_i, dim=0)
        return x_j * ops.expand_dims(alpha, -1)
