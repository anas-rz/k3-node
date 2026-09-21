from typing import Optional, Union, Tuple
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import add_self_loops, remove_self_loops, softmax


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        heads: Number of multi-head-attentions. (default: ``1``)
        concat: If set to :obj:`False`, the multi-head-attentions are averaged
            instead of concatenated. (default: ``True``)
        negative_slope: LeakyReLU angle of the negative slope. (default: ``0.2``)
        dropout: Dropout probability of the normalized attention coefficients.
            (default: ``0.0``)
        add_self_loops: If set to :obj:`False`, will not add self-loops to
            the input graph. (default: ``True``)
        edge_dim: Edge feature dimensionality (in case there are any).
            (default: :obj:`None`)
        fill_value: The way to generate edge features of self-loops
            (default: ``"mean"``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
        share_weights: If set to :obj:`True`, the same matrix will be applied
            to the source and target node features. (default: ``False``)
        residual: If set to :obj:`True`, will compute residual connections.
            (default: ``False``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, str] = "mean",
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.use_bias = bias
        self.share_weights = share_weights
        self.residual = residual

        total_out_channels = out_channels * (heads if concat else 1)

        if isinstance(in_channels, int):
            self.lin = layers.Dense(heads * out_channels, use_bias=False)
            self.lin_src = self.lin
            self.lin_dst = self.lin
        else:
            self.lin = None
            self.lin_src = layers.Dense(heads * out_channels, use_bias=False)
            if share_weights:
                self.lin_dst = self.lin_src
            else:
                self.lin_dst = layers.Dense(heads * out_channels, use_bias=False)

        if edge_dim is not None:
            self.lin_edge = layers.Dense(heads * out_channels, use_bias=False)
        else:
            self.lin_edge = None

        if residual:
            self.res = layers.Dense(total_out_channels, use_bias=False)
        else:
            self.res = None

        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_channels_src = input_shape[0][-1]
            in_channels_dst = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_channels_src
        else:
            in_channels_src = input_shape[-1]
            in_channels_dst = input_shape[-1]

        self.lin_src.build((None, in_channels_src))
        if self.lin_dst is not self.lin_src:
            self.lin_dst.build((None, in_channels_dst))
        if self.lin_edge is not None:
            self.lin_edge.build((None, self.edge_dim))
        if self.res is not None:
            self.res.build((None, in_channels_dst))

        self.att_src = self.add_weight(
            shape=(1, self.heads, self.out_channels),
            initializer="glorot_uniform",
            name="att_src",
        )
        self.att_dst = self.add_weight(
            shape=(1, self.heads, self.out_channels),
            initializer="glorot_uniform",
            name="att_dst",
        )
        if self.edge_dim is not None:
            self.att_edge = self.add_weight(
                shape=(1, self.heads, self.out_channels),
                initializer="glorot_uniform",
                name="att_edge",
            )
        else:
            self.att_edge = None

        total_out_channels = self.out_channels * (self.heads if self.concat else 1)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(total_out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None
        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, size=None, return_attention_weights=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        H, C = self.heads, self.out_channels
        if isinstance(x, (tuple, list)):
            x_src, x_dst = x[0], x[1]
        else:
            x_src, x_dst = x, x

        x_src_proj = ops.reshape(self.lin_src(x_src), (-1, H, C))
        x_dst_proj = ops.reshape(self.lin_dst(x_dst), (-1, H, C)) if x_dst is not None else None

        alpha_src = ops.sum(x_src_proj * self.att_src, axis=-1)
        alpha_dst = ops.sum(x_dst_proj * self.att_dst, axis=-1) if x_dst_proj is not None else None

        if self.add_self_loops:
            if not isinstance(x, (tuple, list)):
                num_nodes = ops.shape(x)[0]
            else:
                num_nodes = ops.shape(x_src)[0]
                if x_dst is not None:
                    num_nodes = ops.minimum(num_nodes, ops.shape(x_dst)[0])
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes
            )

        row, col = edge_index[0], edge_index[1]
        row, col = ops.cast(row, "int32"), ops.cast(col, "int32")

        alpha_j = ops.take(alpha_src, row, axis=0)
        alpha_i = ops.take(alpha_dst, col, axis=0) if alpha_dst is not None else 0.0
        alpha = alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None and self.att_edge is not None:
            edge_attr_proj = ops.reshape(self.lin_edge(edge_attr), (-1, H, C))
            alpha = alpha + ops.sum(edge_attr_proj * self.att_edge, axis=-1)

        alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)
        num_nodes_dst = ops.shape(x_dst)[0] if x_dst is not None else ops.shape(x_src)[0]
        alpha = softmax(alpha, col, num_nodes=num_nodes_dst, dim=0)

        if self.dropout is not None:
            alpha = self.dropout(alpha)

        # Message & aggregate
        x_src_j = ops.take(x_src_proj, row, axis=0)
        out = ops.expand_dims(alpha, -1) * x_src_j
        out = ops.segment_sum(out, col, num_segments=num_nodes_dst)

        if self.concat:
            out = ops.reshape(out, (-1, H * C))
        else:
            out = ops.mean(out, axis=1)

        if self.res is not None and x_dst is not None:
            out = out + self.res(x_dst)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, alpha)
        return out


class FusedGATConv(GATConv):
    r"""The fused graph attentional operator."""
    pass
