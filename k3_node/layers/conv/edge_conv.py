from typing import Callable, Optional, Union, Tuple
import keras
from keras import layers, ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.pool.knn import knn_graph


class EdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.

    Args:
        nn: A neural network :math:`h_{\mathbf{\Theta}}` that maps
            pair-wise node features to new edge representations.
        aggr: The aggregation scheme to use (``"max"``, ``"mean"``, ``"sum"``).
            (default: ``"max"``)
    """

    def __init__(self, nn: Callable, aggr: str = "max", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn

    def build(self, input_shape):
        if hasattr(self.nn, "build") and not getattr(self.nn, "built", False):
            if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
                c = input_shape[0][-1]
            elif isinstance(input_shape, (tuple, list)):
                c = input_shape[-1]
            else:
                c = None
            nn_shape = (None, 2 * c) if c is not None else None
            self.nn.build(nn_shape)
        self.built = True

    def call(self, x, edge_index=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        return self.propagate(edge_index, x=(x_src, x_dst), **kwargs)

    def message(self, x_i, x_j):
        return self.nn(ops.concatenate([x_i, x_j - x_i], axis=-1))


class DynamicEdgeConv(EdgeConv):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper,
    which dynamically constructs a graph using :math:`k`-NN at each layer.

    Args:
        nn: A neural network :math:`h_{\mathbf{\Theta}}`.
        k: Number of nearest neighbors. (default: ``6``)
        aggr: The aggregation scheme to use (``"max"``, ``"mean"``, ``"sum"``).
            (default: ``"max"``)
        num_workers: Number of workers (ignored in Keras backend).
    """

    def __init__(self, nn: Callable, k: int = 6, aggr: str = "max", num_workers: int = 1, **kwargs):
        super().__init__(nn=nn, aggr=aggr, **kwargs)
        self.k = k

    def call(self, x, batch=None, **kwargs):
        if isinstance(x, (tuple, list)):
            x_src = x[0]
        else:
            x_src = x

        edge_index = knn_graph(x_src, k=self.k, batch=batch, loop=False, flow=self.flow)
        return super().call(x, edge_index=edge_index, **kwargs)
