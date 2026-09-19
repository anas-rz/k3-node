import math
from typing import Optional, Union, Tuple
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Meta-Learning on Graph Neural Networks"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        heads: Number of multi-head-attentions. (default: ``1``)
        concat: If set to :obj:`False`, the multi-head-attentions are averaged
            instead of concatenated. (default: ``True``)
        beta: If set to :obj:`True`, will use a gated residual connection.
            (default: ``False``)
        dropout: Dropout probability of the normalized attention coefficients.
            (default: ``0.0``)
        edge_dim: Edge feature dimensionality (in case there are any).
            (default: :obj:`None`)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
        root_weight: If set to :obj:`False`, the layer will not add the
            transformed root node features. (default: ``True``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.use_bias = bias

        total_out_channels = out_channels * (heads if concat else 1)

        self.lin_key = layers.Dense(heads * out_channels, use_bias=bias)
        self.lin_query = layers.Dense(heads * out_channels, use_bias=bias)
        self.lin_value = layers.Dense(heads * out_channels, use_bias=bias)

        if edge_dim is not None:
            self.lin_edge = layers.Dense(heads * out_channels, use_bias=False)
        else:
            self.lin_edge = None

        if root_weight:
            self.lin_skip = layers.Dense(total_out_channels, use_bias=bias)
            if self.beta:
                self.lin_beta = layers.Dense(1, use_bias=False)
            else:
                self.lin_beta = None
        else:
            self.lin_skip = None
            self.lin_beta = None

        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_channels_src = input_shape[0][-1]
            in_channels_dst = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_channels_src
        else:
            in_channels_src = input_shape[-1]
            in_channels_dst = input_shape[-1]

        self.lin_key.build((None, in_channels_src))
        self.lin_query.build((None, in_channels_dst))
        self.lin_value.build((None, in_channels_src))

        if self.lin_edge is not None:
            self.lin_edge.build((None, self.edge_dim))
        if self.lin_skip is not None:
            self.lin_skip.build((None, in_channels_dst))
        if self.lin_beta is not None:
            total_out = self.out_channels * (self.heads if self.concat else 1)
            self.lin_beta.build((None, 3 * total_out))

        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, return_attention_weights=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        H, C = self.heads, self.out_channels
        if isinstance(x, (tuple, list)):
            x_src, x_dst = x[0], x[1]
        else:
            x_src, x_dst = x, x

        query = ops.reshape(self.lin_query(x_dst), (-1, H, C))
        key = ops.reshape(self.lin_key(x_src), (-1, H, C))
        value = ops.reshape(self.lin_value(x_src), (-1, H, C))

        row, col = edge_index[0], edge_index[1]
        row, col = ops.cast(row, "int32"), ops.cast(col, "int32")

        query_i = ops.take(query, col, axis=0)
        key_j = ops.take(key, row, axis=0)
        value_j = ops.take(value, row, axis=0)

        if self.lin_edge is not None and edge_attr is not None:
            edge_attr_proj = ops.reshape(self.lin_edge(edge_attr), (-1, H, C))
            key_j = key_j + edge_attr_proj
            value_j = value_j + edge_attr_proj

        alpha = ops.sum(query_i * key_j, axis=-1) / math.sqrt(C)
        num_nodes_dst = ops.shape(x_dst)[0]
        alpha = softmax(alpha, col, num_nodes=num_nodes_dst, dim=0)

        if self.dropout is not None:
            alpha = self.dropout(alpha)

        out = ops.expand_dims(alpha, -1) * value_j
        out = ops.segment_sum(out, col, num_segments=num_nodes_dst)

        if self.concat:
            out = ops.reshape(out, (-1, H * C))
        else:
            out = ops.mean(out, axis=1)

        if self.root_weight and self.lin_skip is not None:
            x_r = self.lin_skip(x_dst)
            if self.lin_beta is not None:
                b_input = ops.concatenate([out, x_r, out - x_r], axis=-1)
                b = ops.sigmoid(self.lin_beta(b_input))
                out = b * x_r + (1.0 - b) * out
            else:
                out = out + x_r

        if return_attention_weights:
            return out, (edge_index, alpha)
        return out

