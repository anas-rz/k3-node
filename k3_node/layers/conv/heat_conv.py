from typing import Optional
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax
from k3_node.layers.dense.linear import HeteroLinear


class HEATConv(MessagePassing):
    r"""The heterogeneous edge-enhanced graph attentional operator from the
    `"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent
    Trajectory Prediction" <https://arxiv.org/abs/2106.07161>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_node_types (int): The number of node types.
        num_edge_types (int): The number of edge types.
        edge_type_emb_dim (int): The embedding size of edge types.
        edge_dim (int): Edge feature dimensionality.
        edge_attr_emb_dim (int): The embedding size of edge features.
        heads (int, optional): Number of multi-head-attentions. (default: :obj:`1`)
        concat (bool, optional): Whether to concatenate multi-head attention. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle. (default: :obj:`0.2`)
        root_weight (bool, optional): Whether to add root node features. (default: :obj:`True`)
        bias (bool, optional): Whether to learn an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_node_types: int,
        num_edge_types: int,
        edge_type_emb_dim: int,
        edge_dim: int,
        edge_attr_emb_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.edge_type_emb_dim = edge_type_emb_dim
        self.edge_dim = edge_dim
        self.edge_attr_emb_dim = edge_attr_emb_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.root_weight = root_weight
        self.use_bias = bias

        self.hetero_lin = HeteroLinear(in_channels, out_channels, num_node_types, bias=bias)
        self.edge_type_emb = keras.layers.Embedding(num_edge_types, edge_type_emb_dim)
        self.edge_attr_emb = keras.layers.Dense(edge_attr_emb_dim, use_bias=False)
        self.att = keras.layers.Dense(heads, use_bias=False)
        self.lin = keras.layers.Dense(out_channels, use_bias=bias)

    def build(self, input_shape=None):
        if not self.hetero_lin.built and self.in_channels > 0:
            self.hetero_lin.build((None, self.in_channels))
        if not self.edge_type_emb.built:
            self.edge_type_emb.build((None,))
        if not self.edge_attr_emb.built and self.edge_dim > 0:
            self.edge_attr_emb.build((None, self.edge_dim))
        att_dim = 2 * self.out_channels + self.edge_type_emb_dim + self.edge_attr_emb_dim
        if not self.att.built:
            self.att.build((None, att_dim))
        lin_dim = self.out_channels + self.edge_attr_emb_dim
        if not self.lin.built:
            self.lin.build((None, lin_dim))
        super().build(input_shape)

    def call(self, x, edge_index, node_type, edge_type, edge_attr=None):
        if not self.built:
            self.build()

        x = self.hetero_lin(x, node_type)
        edge_type_emb = ops.leaky_relu(
            self.edge_type_emb(edge_type),
            negative_slope=self.negative_slope,
        )

        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb, edge_attr=edge_attr)

        if self.concat:
            if self.root_weight:
                out = out + ops.expand_dims(x, axis=1)
            out = ops.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = ops.mean(out, axis=1)
            if self.root_weight:
                out = out + x

        return out

    def message(self, x_i, x_j, edge_type_emb, edge_attr, index=None):
        edge_attr = ops.leaky_relu(self.edge_attr_emb(edge_attr), negative_slope=self.negative_slope)
        alpha = ops.concatenate([x_i, x_j, edge_type_emb, edge_attr], axis=-1)
        alpha = ops.leaky_relu(self.att(alpha), negative_slope=self.negative_slope)
        alpha = softmax(alpha, index=index)

        feat = self.lin(ops.concatenate([x_j, edge_attr], axis=-1))
        # feat: (E, out_channels), alpha: (E, heads)
        out = ops.expand_dims(feat, axis=1) * ops.expand_dims(alpha, axis=-1)
        return out

