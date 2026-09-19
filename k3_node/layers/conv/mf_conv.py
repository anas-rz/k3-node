from typing import Optional, Union, Tuple
import keras
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import degree


class MFConv(MessagePassing):
    r"""The molecular fingerprint graph convolutional operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`_ paper.

    Args:
        in_channels: Size of each input sample, or a tuple for bipartite graphs.
        out_channels: Size of each output sample.
        max_degree: The maximum degree of any node. (default: ``10``)
        bias: If set to :obj:`False`, the layer will not learn an additive bias.
            (default: ``True``)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        max_degree: int = 10,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree
        self.use_bias = bias

        self.lins_l = [layers.Dense(out_channels, use_bias=bias) for _ in range(max_degree + 1)]
        self.lins_r = [layers.Dense(out_channels, use_bias=False) for _ in range(max_degree + 1)]

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            in_channels_src = input_shape[0][-1]
            in_channels_dst = input_shape[1][-1] if len(input_shape) > 1 and input_shape[1] is not None else in_channels_src
        else:
            in_channels_src = input_shape[-1]
            in_channels_dst = input_shape[-1]

        for lin_l in self.lins_l:
            lin_l.build((None, in_channels_src))
        for lin_r in self.lins_r:
            lin_r.build((None, in_channels_dst))
        self.built = True

    def call(self, x, edge_index=None, size=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        target_idx = edge_index[1] if self.flow == "source_to_target" else edge_index[0]
        N = ops.shape(x_dst)[self.node_dim] if x_dst is not None else ops.shape(x_src)[self.node_dim]
        deg = degree(target_idx, num_nodes=N)
        deg = ops.clip(deg, 0, self.max_degree)

        h = self.propagate(edge_index, x=(x_src, x_dst), size=size)

        out = ops.zeros((ops.shape(h)[0], self.out_channels), dtype=h.dtype)
        for i, (lin_l, lin_r) in enumerate(zip(self.lins_l, self.lins_r)):
            mask = ops.equal(deg, i)
            mask = ops.expand_dims(ops.cast(mask, h.dtype), -1)
            term = lin_l(h)
            if x_dst is not None:
                term = term + lin_r(x_dst)
            out = out + mask * term

        return out

    def message(self, x_j):
        return x_j

