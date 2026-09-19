from typing import Optional, Union, Tuple, List, Callable
import keras
from keras import ops
from keras.layers import Dense, BatchNormalization, LayerNormalization

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.aggr import SoftmaxAggregation, PowerMeanAggregation


class GENConv(MessagePassing):
    r"""The generalized graph convolution operator from the `"DeeperGCN: All
    You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "softmax",
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: Optional[str] = "batch",
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        if aggr in ("softmax", "softmax_sg"):
            aggr_module = SoftmaxAggregation(t=t, learn=learn_t)
        elif aggr in ("power", "powermean"):
            aggr_module = PowerMeanAggregation(p=p, learn=learn_p)
        else:
            aggr_module = aggr

        super().__init__(aggr=aggr_module, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self.edge_dim = edge_dim
        self.use_bias = bias

        if isinstance(in_channels, int):
            self.in_channels_l = in_channels
            self.in_channels_r = in_channels
        else:
            self.in_channels_l, self.in_channels_r = in_channels

        if self.in_channels_l != out_channels:
            self.lin_src = Dense(out_channels, use_bias=bias)
        else:
            self.lin_src = None

        if edge_dim is not None and edge_dim != out_channels:
            self.lin_edge = Dense(out_channels, use_bias=bias)
        else:
            self.lin_edge = None

        if self.in_channels_r != out_channels:
            self.lin_dst = Dense(out_channels, use_bias=bias)
        else:
            self.lin_dst = None

        # MLP
        self.mlp_layers = []
        channels = [out_channels]
        for _ in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)

        for i in range(len(channels) - 1):
            self.mlp_layers.append(Dense(channels[i + 1], use_bias=bias))
            if i < len(channels) - 2:
                if norm == "batch":
                    self.mlp_layers.append(BatchNormalization())
                elif norm == "layer":
                    self.mlp_layers.append(LayerNormalization())
                self.mlp_layers.append(keras.layers.ReLU())

    def build(self, input_shape=None):
        if self.lin_src is not None:
            self.lin_src.build((None, self.in_channels_l))
        if self.lin_dst is not None:
            self.lin_dst.build((None, self.in_channels_r))
        if self.lin_edge is not None:
            self.lin_edge.build((None, self.edge_dim))
        curr_dim = self.out_channels
        for layer in self.mlp_layers:
            if hasattr(layer, "build"):
                layer.build((None, curr_dim))
            if hasattr(layer, "units"):
                curr_dim = layer.units
        self.built = True

    def call(self, inputs, edge_index=None, edge_attr=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_attr = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_l, x_r = x
        else:
            x_l = x_r = x

        if self.lin_src is not None:
            x_l = self.lin_src(x_l)

        num_nodes = ops.shape(x_r)[0]
        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            size=(ops.shape(x_l)[0], num_nodes),
        )

        x_dst = x_r
        if self.lin_dst is not None:
            x_dst = self.lin_dst(x_dst)
        out = out + x_dst

        for layer in self.mlp_layers:
            out = layer(out)

        return out

    def message(self, x_j, edge_attr=None):
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
        msg = x_j if edge_attr is None else x_j + edge_attr
        return ops.relu(msg) + self.eps

