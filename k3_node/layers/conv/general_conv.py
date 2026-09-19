import math
from typing import Optional, Union, Tuple
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import softmax


class GeneralConv(MessagePassing):
    r"""A general GNN layer adapted from the `"Design Space for Graph Neural
    Networks" <https://arxiv.org/abs/2011.08843>`_ paper.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: Optional[int] = None,
        in_edge_channels: Optional[int] = None,
        aggr: str = "add",
        skip_linear: bool = False,
        directed_msg: bool = True,
        heads: int = 1,
        attention: bool = False,
        attention_type: str = "additive",
        l2_normalize: bool = False,
        bias: bool = True,
        # Spektral compatibility arguments:
        channels: Optional[int] = None,
        batch_norm: Optional[bool] = None,
        dropout: Optional[float] = None,
        aggregate: Optional[str] = None,
        activation: Optional[str] = None,
        use_bias: Optional[bool] = None,
        **kwargs,
    ):
        if out_channels is None:
            if channels is not None:
                out_channels = channels
                in_channels = -1
            else:
                out_channels = in_channels
                in_channels = -1

        if aggregate is not None:
            aggr = aggregate
        if use_bias is not None:
            bias = use_bias

        kwargs.setdefault("aggr", aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels
        self.skip_linear = skip_linear
        self.directed_msg = directed_msg
        self.heads = heads
        self.attention = attention
        self.attention_type = attention_type
        self.l2_normalize = l2_normalize
        self.use_bias = bias

        if isinstance(in_channels, int):
            self.in_channels_l = in_channels
            self.in_channels_r = in_channels
        else:
            self.in_channels_l, self.in_channels_r = in_channels

        self.lin_msg = Dense(out_channels * heads, use_bias=bias)
        if not directed_msg:
            self.lin_msg_i = Dense(out_channels * heads, use_bias=bias)
        else:
            self.lin_msg_i = None

        if skip_linear or self.in_channels_r != out_channels:
            self.lin_self = Dense(out_channels, use_bias=bias)
        else:
            self.lin_self = None

        if in_edge_channels is not None:
            self.lin_edge = Dense(out_channels * heads, use_bias=bias)
        else:
            self.lin_edge = None

        if attention:
            if attention_type == "additive":
                self.att_msg = self.add_weight(
                    shape=(1, heads, out_channels),
                    initializer="glorot_uniform",
                    name="att_msg",
                )
            elif attention_type == "dot_product":
                self.scaler = math.sqrt(out_channels)
            else:
                raise ValueError(f"Attention type '{attention_type}' not supported")
        else:
            self.att_msg = None

    def build(self, input_shape=None):
        if input_shape is not None:
            if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                dim = input_shape[0][-1]
            elif isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and input_shape[0] is not None and not isinstance(input_shape[0], (int, type(None))):
                dim = getattr(input_shape[0], "shape", [None, None])[-1]
            else:
                dim = input_shape[-1]
            if (self.in_channels_l is None or self.in_channels_l == -1) and dim is not None:
                self.in_channels_l = self.in_channels_r = dim
        if self.in_channels_l is not None and self.in_channels_l != -1:
            self.lin_msg.build((None, self.in_channels_l))
            if self.lin_msg_i is not None:
                self.lin_msg_i.build((None, self.in_channels_r))
            if self.lin_self is not None:
                self.lin_self.build((None, self.in_channels_r))
            if self.lin_edge is not None:
                self.lin_edge.build((None, self.in_edge_channels))
        self.built = True

    def call(self, inputs, edge_index=None, edge_attr=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_attr = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected inputs length {len(inputs)}")
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if isinstance(x, (list, tuple)):
            x_l, x_r = x
        else:
            x_l = x_r = x

        if self.in_channels_l is None or self.in_channels_l == -1:
            dim = ops.shape(x_l)[-1]
            self.in_channels_l = self.in_channels_r = dim
            self.build((None, dim))

        # Check legacy adj matrix
        is_legacy = False
        if hasattr(edge_index, "shape") and len(edge_index.shape) == 2:
            if edge_index.shape[0] != 2 and edge_index.shape[0] == edge_index.shape[1]:
                is_legacy = True
        elif not hasattr(edge_index, "shape"):
            is_legacy = True

        if is_legacy:
            if hasattr(edge_index, "indices"):
                edge_index = ops.transpose(edge_index.indices)
            else:
                adj = edge_index
                row, col = ops.where(adj > 0)
                edge_index = ops.stack([row, col], axis=0)

        num_nodes = ops.shape(x_r)[0]
        size = (ops.shape(x_l)[0], num_nodes)

        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            size=size,
        )
        out = ops.mean(out, axis=1)  # aggregate heads

        if self.lin_self is not None:
            out = out + self.lin_self(x_r)
        else:
            out = out + x_r

        if self.l2_normalize:
            out = out / (ops.norm(out, axis=-1, keepdims=True) + 1e-12)

        return out

    def _message_basic(self, x_i, x_j, edge_attr=None):
        if self.directed_msg:
            x_j = self.lin_msg(x_j)
        else:
            x_j = self.lin_msg(x_j) + self.lin_msg_i(x_i)
        if edge_attr is not None and self.lin_edge is not None:
            x_j = x_j + self.lin_edge(edge_attr)
        return x_j

    def message(self, x_i, x_j, edge_attr=None, index=None, size_i=None):
        x_j_out = self._message_basic(x_i, x_j, edge_attr)
        x_j_out = ops.reshape(x_j_out, (-1, self.heads, self.out_channels))

        if self.attention:
            if self.attention_type == "dot_product":
                x_i_out = self._message_basic(x_j, x_i, edge_attr)
                x_i_out = ops.reshape(x_i_out, (-1, self.heads, self.out_channels))
                alpha = ops.sum(x_i_out * x_j_out, axis=-1) / self.scaler
            else:
                alpha = ops.sum(x_j_out * self.att_msg, axis=-1)
            alpha = ops.leaky_relu(alpha, negative_slope=0.2)
            alpha = softmax(alpha, index, num_nodes=size_i, dim=0)
            return x_j_out * ops.expand_dims(alpha, -1)
        else:
            return x_j_out
