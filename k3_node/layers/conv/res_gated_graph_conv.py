from typing import Callable, Optional, Union, Tuple
import keras
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing


class ResGatedGraphConv(MessagePassing):
    r"""The residual gated graph convolutional operator from the
    `"Residual Gated Graph ConvNets" <https://arxiv.org/abs/1711.07553>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        act: Activation function :math:`\sigma` for gating. (default: ``"sigmoid"``)
        edge_dim: Edge feature dimensionality. (default: :obj:`None`)
        root_weight: If set to :obj:`False`, will not add transformed root features.
            (default: ``True``)
        bias: If set to :obj:`False`, will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        act: Union[str, Callable] = "sigmoid",
        edge_dim: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = keras.activations.get(act) if isinstance(act, str) else act
        self.edge_dim = edge_dim
        self.root_weight = root_weight
        self.use_bias = bias

        in_src = in_channels[0] if isinstance(in_channels, (tuple, list)) else in_channels
        in_dst = in_channels[1] if isinstance(in_channels, (tuple, list)) else in_channels
        e_dim = edge_dim if edge_dim is not None else 0

        self.lin_key = layers.Dense(out_channels, use_bias=True)
        self.lin_query = layers.Dense(out_channels, use_bias=True)
        self.lin_value = layers.Dense(out_channels, use_bias=True)

        if root_weight:
            self.lin_skip = layers.Dense(out_channels, use_bias=False)
        else:
            self.lin_skip = None

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_src = input_shape[0][-1]
            in_dst = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_src
        else:
            in_src = input_shape[-1]
            in_dst = input_shape[-1]

        e_dim = self.edge_dim if self.edge_dim is not None else 0
        self.lin_key.build((None, in_dst + e_dim))
        self.lin_query.build((None, in_src + e_dim))
        self.lin_value.build((None, in_src + e_dim))

        if self.lin_skip is not None:
            self.lin_skip.build((None, in_dst))

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None
        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        if self.edge_dim is None:
            k = self.lin_key(x_dst)
            q = self.lin_query(x_src)
            v = self.lin_value(x_src)
        else:
            k, q, v = x_dst, x_src, x_src

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        if self.root_weight and self.lin_skip is not None and x_dst is not None:
            out = out + self.lin_skip(x_dst)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, k_i, q_j, v_j, edge_attr=None):
        if self.edge_dim is not None and edge_attr is not None:
            k_i = self.lin_key(ops.concatenate([k_i, edge_attr], axis=-1))
            q_j = self.lin_query(ops.concatenate([q_j, edge_attr], axis=-1))
            v_j = self.lin_value(ops.concatenate([v_j, edge_attr], axis=-1))

        gate = self.act(k_i + q_j)
        return gate * v_j

