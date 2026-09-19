from typing import Callable, Union, Tuple
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing


class NNConv(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        nn: A neural network :math:`h_{\mathbf{\Theta}}` that maps edge features
            to shape :obj:`[-1, in_channels * out_channels]`.
        aggr: The aggregation scheme to use (``"add"``, ``"mean"``, ``"max"``).
            (default: ``"add"``)
        root_weight: If set to :obj:`False`, the layer will not add the
            transformed root node features. (default: ``True``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        nn: Callable,
        aggr: str = "add",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.root_weight = root_weight
        self.use_bias = bias

        if isinstance(in_channels, int):
            self.in_channels_src = in_channels
            self.in_channels_dst = in_channels
        else:
            self.in_channels_src = in_channels[0]
            self.in_channels_dst = in_channels[1]

        if root_weight:
            self.lin_root = layers.Dense(out_channels, use_bias=False)
        else:
            self.lin_root = None

    def build(self, input_shape):
        if self.lin_root is not None:
            self.lin_root.build((None, self.in_channels_dst))

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None
        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, size=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        weight = self.nn(edge_attr)
        weight = ops.reshape(weight, (-1, self.in_channels_src, self.out_channels))

        out = self.propagate(edge_index, x=x_src, weight=weight, size=size)

        if self.root_weight and self.lin_root is not None and x_dst is not None:
            out = out + self.lin_root(x_dst)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, weight):
        # x_j: [E, in_channels_src], weight: [E, in_channels_src, out_channels]
        x_j = ops.expand_dims(x_j, axis=1)  # [E, 1, in_channels_src]
        msg = ops.matmul(x_j, weight)       # [E, 1, out_channels]
        return ops.squeeze(msg, axis=1)     # [E, out_channels]

