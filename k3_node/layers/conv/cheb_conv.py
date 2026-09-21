from typing import Optional, List
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import get_laplacian


class ChebConv(MessagePassing):
    r"""The Chebyshev spectral graph convolutional operator from the
    `"Convolutional Neural Networks on Graphs with Fast Localized Spectral
    Filtering" <https://arxiv.org/abs/1606.09375>`_ paper.

    Args:
        in_channels: Size of each input sample.
        out_channels: Size of each output sample.
        K: Chebyshev filter size :math:`K`.
        normalization: The normalization scheme for the graph
            Laplacian (``"sym"``, ``"rw"`` or :obj:`None`). (default: ``"sym"``)
        bias: If set to :obj:`False`, the layer will not learn
            an additive bias. (default: ``"True"``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = "sym",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        if K <= 0:
            raise ValueError(f"K must be a positive integer, got {K}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.use_bias = bias

        self.lins = [layers.Dense(out_channels, use_bias=False) for _ in range(K)]

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
            feat_shape = input_shape[0]
        else:
            feat_shape = input_shape
        for lin in self.lins:
            lin.build(feat_shape)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None
        self.built = True

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight=None,
        normalization: Optional[str] = "sym",
        lambda_max=None,
        dtype=None,
    ):
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes
        )
        if lambda_max is None:
            lambda_max = 2.0 * ops.max(edge_weight)
        else:
            lambda_max = ops.convert_to_tensor(lambda_max, dtype=edge_weight.dtype)

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight = ops.where(
            ops.isinf(edge_weight) | ops.isnan(edge_weight), 0.0, edge_weight
        )

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight = ops.where(loop_mask, edge_weight - 1.0, edge_weight)
        return edge_index, edge_weight

    def call(self, x, edge_index=None, edge_weight=None, lambda_max=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        num_nodes = x.shape[self.node_dim] if hasattr(x, "shape") and x.shape[self.node_dim] is not None else ops.shape(x)[self.node_dim]
        edge_index, norm = self.__norm__(
            edge_index,
            num_nodes,
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
        )

        Tx_0 = x
        Tx_1 = x
        out = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, norm=None):
        if norm is None:
            return x_j
        return ops.expand_dims(norm, -1) * x_j

