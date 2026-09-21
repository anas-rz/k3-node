import math
import keras
from keras import ops
from k3_node.layers.conv.gcn_conv import gcn_norm
from k3_node.layers.conv.message_passing import MessagePassing


def restricted_softmax(src, axis: int = -1, margin: float = 0.0):
    src_max = ops.maximum(ops.max(src, axis=axis, keepdims=True), 0.0)
    out = ops.exp(src - src_max)
    denom = ops.sum(out, axis=axis, keepdims=True) + ops.exp(margin - src_max)
    return out / denom


class GroupedLinear(keras.layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.use_bias = bias

    def build(self, input_shape=None):
        self.weight = self.add_weight(
            shape=(self.groups, self.in_channels // self.groups, self.out_channels // self.groups),
            initializer="glorot_uniform",
            trainable=True,
            name="weight",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, src):
        if self.groups > 1:
            orig_shape = ops.shape(src)
            # Flatten batch dims
            src_flat = ops.reshape(src, (-1, self.groups, self.in_channels // self.groups))
            src_trans = ops.transpose(src_flat, (1, 0, 2))
            out = ops.matmul(src_trans, self.weight)
            out = ops.transpose(out, (1, 0, 2))
            out_shape = tuple(orig_shape[:-1]) + (self.out_channels,)
            out = ops.reshape(out, out_shape)
        else:
            out = ops.matmul(src, self.weight[0])

        if self.bias is not None:
            out = out + self.bias
        return out


class DNAMultiHead(keras.layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, groups: int = 1, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.use_bias = bias

        self.lin_q = GroupedLinear(in_channels, out_channels, groups, bias)
        self.lin_k = GroupedLinear(in_channels, out_channels, groups, bias)
        self.lin_v = GroupedLinear(in_channels, out_channels, groups, bias)

    def build(self, input_shape=None):
        self.lin_q.build()
        self.lin_k.build()
        self.lin_v.build()
        super().build(input_shape)

    def call(self, query, key, value):
        q = self.lin_q(query)
        k = self.lin_k(key)
        v = self.lin_v(value)

        # q: (E, 1, C) -> (E, heads, 1, C // heads)
        E = ops.shape(q)[0]
        H = self.heads
        D = self.out_channels // H

        q_entries = ops.shape(q)[1]
        k_entries = ops.shape(k)[1]

        q = ops.transpose(ops.reshape(q, (E, q_entries, H, D)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, (E, k_entries, H, D)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, (E, k_entries, H, D)), (0, 2, 1, 3))

        # score: (E, H, q_entries, k_entries)
        score = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / math.sqrt(D)
        score = restricted_softmax(score, axis=-1)

        out = ops.matmul(score, v)  # (E, H, q_entries, D)
        out = ops.transpose(out, (0, 2, 1, 3))  # (E, q_entries, H, D)
        return ops.reshape(out, (E, q_entries, self.out_channels))


class DNAConv(MessagePassing):
    r"""The dynamic neighborhood aggregation operator from the `"Just Jump:
    Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
    <https://arxiv.org/abs/1904.04849>`_ paper.

    Args:
        channels (int): Size of each input/output sample.
        heads (int, optional): Number of multi-head-attentions. (default: :obj:`1`)
        groups (int, optional): Number of groups for linear projections. (default: :obj:`1`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
        cached (bool, optional): Whether to cache GCN normalization. (default: :obj:`False`)
        normalize (bool, optional): Whether to apply symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): Whether to add self-loops. (default: :obj:`True`)
        bias (bool, optional): Whether to learn an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.channels = channels
        self.heads = heads
        self.groups = groups
        self.dropout = dropout
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.use_bias = bias

        self.multi_head = DNAMultiHead(channels, channels, heads, groups, bias)

    def build(self, input_shape=None):
        self.multi_head.build()
        super().build(input_shape)

    def call(self, x, edge_index, edge_weight=None):
        if not self.built:
            self.build()

        if len(ops.shape(x)) == 2:
            x = ops.expand_dims(x, axis=1)

        num_nodes = ops.shape(x)[0]
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                num_nodes=num_nodes,
                add_self_loops=self.add_self_loops,
                dtype=x.dtype,
            )

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight=None):
        q = x_i[:, -1:]  # (E, 1, C)
        out = self.multi_head(q, x_j, x_j)  # (E, 1, C)
        out = ops.squeeze(out, axis=1)  # (E, C)
        if edge_weight is not None:
            out = ops.expand_dims(edge_weight, axis=-1) * out
        return out

