from typing import Optional
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import scatter, softmax


class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): Whether to use hypergraph attention. (default: :obj:`False`)
        attention_mode (str, optional): Attention mode (:obj:`"node"` or :obj:`"edge"`). (default: :obj:`"node"`)
        heads (int, optional): Number of multi-head-attentions. (default: :obj:`1`)
        concat (bool, optional): Whether to concatenate heads. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle. (default: :obj:`0.2`)
        bias (bool, optional): Whether to learn an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = "node",
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        assert attention_mode in ["node", "edge"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode
        self.negative_slope = negative_slope
        self.use_bias = bias

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.lin = keras.layers.Dense(heads * out_channels, use_bias=False)
        else:
            self.heads = 1
            self.concat = True
            self.lin = keras.layers.Dense(out_channels, use_bias=False)

    def build(self, input_shape=None):
        if not self.lin.built and self.in_channels > 0:
            self.lin.build((None, self.in_channels))

        if self.use_attention:
            self.att = self.add_weight(
                shape=(1, self.heads, 2 * self.out_channels),
                initializer="glorot_uniform",
                trainable=True,
                name="att",
            )

        out_dim = self.heads * self.out_channels if self.concat else self.out_channels
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(out_dim,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(
        self,
        x,
        hyperedge_index,
        hyperedge_weight=None,
        hyperedge_attr=None,
        num_edges=None,
    ):
        if not self.built:
            self.build((None, self.in_channels))

        num_nodes = ops.shape(x)[0]
        if num_edges is None:
            num_edges = 0
            if ops.shape(hyperedge_index)[1] > 0:
                num_edges = int(ops.max(hyperedge_index[1])) + 1

        if hyperedge_weight is None:
            hyperedge_weight = ops.ones((num_edges,), dtype=x.dtype)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = ops.reshape(x, (-1, self.heads, self.out_channels))
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = ops.reshape(hyperedge_attr, (-1, self.heads, self.out_channels))

            x_i = ops.take(x, hyperedge_index[0], axis=0)
            x_j = ops.take(hyperedge_attr, hyperedge_index[1], axis=0)

            alpha = ops.sum(ops.concatenate([x_i, x_j], axis=-1) * self.att, axis=-1)
            alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)

            if self.attention_mode == "node":
                alpha = softmax(alpha, index=hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, index=hyperedge_index[0], num_nodes=num_nodes)

        edge_w = ops.take(hyperedge_weight, hyperedge_index[1], axis=0)
        D = scatter(edge_w, hyperedge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        D = ops.where(ops.equal(D, 0), 0.0, 1.0 / D)

        ones_e = ops.ones((ops.shape(hyperedge_index)[1],), dtype=x.dtype)
        B = scatter(ones_e, hyperedge_index[1], dim=0, dim_size=num_edges, reduce="sum")
        B = ops.where(ops.equal(B, 0), 0.0, 1.0 / B)

        # 1. nodes -> hyperedges
        out = self.propagate(
            hyperedge_index,
            x=x,
            norm=B,
            alpha=alpha,
            size=(num_nodes, num_edges),
        )
        # 2. hyperedges -> nodes
        flipped_edge_index = ops.stack([hyperedge_index[1], hyperedge_index[0]], axis=0)
        out = self.propagate(
            flipped_edge_index,
            x=out,
            norm=D,
            alpha=alpha,
            size=(num_edges, num_nodes),
        )

        if self.concat:
            out = ops.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = ops.mean(out, axis=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm_i, alpha=None):
        H, F = self.heads, self.out_channels
        norm_i_exp = ops.expand_dims(ops.expand_dims(norm_i, axis=-1), axis=-1)
        out = norm_i_exp * ops.reshape(x_j, (-1, H, F))
        if alpha is not None:
            out = ops.expand_dims(alpha, axis=-1) * out
        return out

