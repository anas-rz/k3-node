from typing import Optional
import keras
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax


class RGATConv(MessagePassing):
    r"""The relational graph attentional operator from the
    `"Relational Graph Attention Networks" <https://arxiv.org/abs/1904.05811>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        mod: Optional[str] = None,
        attention_mechanism: str = "across-relation",
        attention_mode: str = "additive-self-attention",
        heads: int = 1,
        dim: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.mod = mod
        self.attention_mechanism = attention_mechanism
        self.attention_mode = attention_mode
        self.heads = heads
        self.dim = dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self.edge_dim = edge_dim
        self.use_bias = bias

        self.q = self.add_weight(
            shape=(heads * out_channels, heads * dim),
            initializer="glorot_uniform",
            name="q",
        )
        self.k = self.add_weight(
            shape=(heads * out_channels, heads * dim),
            initializer="glorot_uniform",
            name="k",
        )
        if edge_dim is not None:
            self.lin_edge = Dense(heads * out_channels, use_bias=False)
            self.e = self.add_weight(
                shape=(heads * out_channels, heads * dim),
                initializer="glorot_uniform",
                name="e",
            )
        else:
            self.lin_edge = None
            self.e = None

        if num_bases is not None:
            self.att = self.add_weight(
                shape=(num_relations, num_bases),
                initializer="glorot_uniform",
                name="att",
            )
            self.basis = self.add_weight(
                shape=(num_bases, in_channels, heads * out_channels),
                initializer="glorot_uniform",
                name="basis",
            )
        elif num_blocks is not None:
            assert (
                in_channels % num_blocks == 0
                and (heads * out_channels) % num_blocks == 0
            )
            self.weight = self.add_weight(
                shape=(
                    num_relations,
                    num_blocks,
                    in_channels // num_blocks,
                    (heads * out_channels) // num_blocks,
                ),
                initializer="glorot_uniform",
                name="weight",
            )
        else:
            self.weight = self.add_weight(
                shape=(num_relations, in_channels, heads * out_channels),
                initializer="glorot_uniform",
                name="weight",
            )

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
        if self.lin_edge is not None and self.edge_dim is not None:
            self.lin_edge.build((None, self.edge_dim))
        self.built = True

    def _get_weight(self):
        if self.num_bases is not None:
            b_flat = ops.reshape(self.basis, (self.num_bases, -1))
            w = ops.matmul(self.att, b_flat)
            return ops.reshape(
                w, (self.num_relations, self.in_channels, self.heads * self.out_channels)
            )
        return self.weight

    def call(
        self,
        inputs,
        edge_index=None,
        edge_type=None,
        edge_attr=None,
        **kwargs,
    ):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 4:
                    x, edge_index, edge_type, edge_attr = inputs
                elif len(inputs) == 3:
                    x, edge_index, edge_type = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError("Unexpected inputs length")
            else:
                raise ValueError("Expected (x, edge_index, ...)")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_l, x_r = x
        else:
            x_l = x_r = x

        edge_index = ops.cast(edge_index, "int32")
        if edge_type is not None:
            edge_type = ops.cast(edge_type, "int32")

        num_nodes = ops.shape(x_r)[0]
        size = (ops.shape(x_l)[0], num_nodes)

        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            edge_type=edge_type,
            edge_attr=edge_attr,
            size=size,
        )

        if self.concat:
            out = ops.reshape(out, (-1, self.heads * self.out_channels))
        else:
            out = ops.mean(out, axis=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_i, x_j, index=None, edge_type=None, edge_attr=None, size_i=None):
        weight = self._get_weight()

        if edge_type is None:
            edge_type = ops.zeros((ops.shape(index)[0],), dtype="int32")

        if self.num_blocks is not None:
            w_r = ops.take(weight, edge_type, axis=0)  # (E, num_blocks, in_b, out_b)
            x_i_b = ops.reshape(
                x_i,
                (-1, self.num_blocks, 1, self.in_channels // self.num_blocks),
            )
            x_j_b = ops.reshape(
                x_j,
                (-1, self.num_blocks, 1, self.in_channels // self.num_blocks),
            )
            outi = ops.reshape(ops.matmul(x_i_b, w_r), (-1, self.heads * self.out_channels))
            outj = ops.reshape(ops.matmul(x_j_b, w_r), (-1, self.heads * self.out_channels))
        else:
            w_r = ops.take(weight, edge_type, axis=0)  # (E, in_channels, heads * out_channels)
            outi = ops.squeeze(ops.matmul(ops.expand_dims(x_i, 1), w_r), 1)
            outj = ops.squeeze(ops.matmul(ops.expand_dims(x_j, 1), w_r), 1)

        qi = ops.matmul(outi, self.q)
        kj = ops.matmul(outj, self.k)

        alpha_edge, alpha = 0.0, 0.0
        if edge_attr is not None and self.lin_edge is not None:
            edge_attributes = ops.reshape(self.lin_edge(edge_attr), (-1, self.heads * self.out_channels))
            alpha_edge = ops.matmul(edge_attributes, self.e)

        if self.attention_mode == "additive-self-attention":
            alpha = qi + kj
            if edge_attr is not None and self.lin_edge is not None:
                alpha = alpha + alpha_edge
            alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)
        elif self.attention_mode == "multiplicative-self-attention":
            alpha = qi * kj
            if edge_attr is not None and self.lin_edge is not None:
                alpha = alpha * alpha_edge

        alpha = softmax(alpha, index, num_nodes=size_i, dim=0)

        outj_h = ops.reshape(outj, (-1, self.heads, self.out_channels))
        if self.attention_mode == "additive-self-attention":
            alpha_h = ops.reshape(alpha, (-1, self.heads, 1))
            return outj_h * alpha_h
        else:
            alpha_h = ops.reshape(alpha, (-1, self.heads, self.dim, 1))
            outj_exp = ops.expand_dims(outj_h, -2)
            msg = alpha_h * outj_exp
            return ops.mean(msg, axis=2)
