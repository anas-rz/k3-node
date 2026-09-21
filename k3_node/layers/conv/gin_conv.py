from typing import Callable, Optional, Union, List
import keras
from keras import layers, ops, activations

from k3_node.layers.conv.message_passing import MessagePassing


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are Graph
    Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        nn: A neural network :math:`h_{\mathbf{\Theta}}` that maps node
            features to new embeddings (e.g. a :class:`keras.Sequential` or callable).
            Also accepts an integer `channels` for backward compatibility.
        eps: (Initial) :math:`\epsilon`-value. (default: ``0.0``)
        train_eps: If set to :obj:`True`, :math:`\epsilon` will be a learnable
            parameter. (default: ``False``)
    """

    def __init__(
        self,
        nn: Union[Callable, int],
        eps: float = 0.0,
        train_eps: bool = False,
        epsilon: Optional[float] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_activation: str = "relu",
        mlp_batchnorm: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=kwargs.pop("aggr", kwargs.pop("aggregate", "add")), **kwargs)

        if epsilon is not None:
            eps = epsilon

        self.initial_eps = eps
        self.train_eps = train_eps

        # Backward compatibility if nn is int (channels)
        if isinstance(nn, int):
            channels = nn
            mlp_hidden = mlp_hidden or []
            act = activations.get(mlp_activation)
            seq_layers = []
            for h in mlp_hidden:
                seq_layers.append(layers.Dense(h, activation=act))
                if mlp_batchnorm:
                    seq_layers.append(layers.BatchNormalization())
            seq_layers.append(layers.Dense(channels, activation=kwargs.get("activation", None)))
            self.nn = keras.Sequential(seq_layers)
        else:
            self.nn = nn

        if train_eps:
            self.eps = self.add_weight(
                shape=(1,),
                initializer=keras.initializers.Constant(eps),
                name="eps",
            )
        else:
            self.eps = ops.cast(eps, "float32")

    def build(self, input_shape):
        if hasattr(self.nn, "build") and not getattr(self.nn, "built", False):
            self.nn.build(input_shape)
        self.built = True

    def call(self, x, edge_index=None, size=None, **kwargs):
        # Handle legacy calling: conv((x, adj))
        if edge_index is None and isinstance(x, (tuple, list)) and len(x) == 2:
            arg0, arg1 = x[0], x[1]
            s1 = getattr(arg1, "shape", None)
            if (
                s1 is not None
                and len(s1) == 2
                and s1[0] is not None
                and s1[1] is not None
                and s1[0] > 2
                and s1[0] == s1[1]
            ):
                where_adj = ops.where(arg1 != 0)
                where_adj = where_adj if not isinstance(where_adj, list) else where_adj
                edge_index = ops.stack([where_adj[0], where_adj[1]], axis=0)
                x = arg0
            elif s1 is not None and len(s1) >= 1 and s1[0] == 2:
                edge_index = arg1
                x = arg0

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        out = self.propagate(edge_index, x=(x_src, x_dst), size=size)

        if x_dst is not None:
            out = out + (1.0 + self.eps) * x_dst

        return self.nn(out)

    def message(self, x_j):
        return x_j


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper, which is able to incorporate edge features into aggregation.

    Args:
        nn: A neural network :math:`h_{\mathbf{\Theta}}`.
        eps: (Initial) :math:`\epsilon`-value. (default: ``0.0``)
        train_eps: If set to :obj:`True`, :math:`\epsilon` will be a learnable
            parameter. (default: ``False``)
        edge_dim: Edge feature dimensionality. (default: :obj:`None`)
    """

    def __init__(
        self,
        nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(aggr=kwargs.pop("aggr", kwargs.pop("aggregate", "add")), **kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.train_eps = train_eps
        self.edge_dim = edge_dim

        if train_eps:
            self.eps = self.add_weight(
                shape=(1,),
                initializer=keras.initializers.Constant(eps),
                name="eps",
            )
        else:
            self.eps = ops.cast(eps, "float32")

        if edge_dim is not None:
            self.lin = layers.Dense(edge_dim, use_bias=True)
        else:
            self.lin = None

    def build(self, input_shape):
        if self.lin is not None:
            if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0 and isinstance(input_shape[0], (tuple, list)):
                node_dim = input_shape[0][-1]
            elif isinstance(input_shape, (tuple, list)):
                node_dim = input_shape[-1]
            else:
                node_dim = 16
            self.lin.units = node_dim
            self.lin.build((None, self.edge_dim))
        self.built = True

    def call(self, x, edge_index=None, edge_attr=None, size=None, **kwargs):
        if edge_index is None and isinstance(x, (tuple, list)):
            x, edge_index = x[0], x[1]

        if not isinstance(x, (tuple, list)):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x[0], x[1]

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, size=size)

        if x_dst is not None:
            out = out + (1.0 + self.eps) * x_dst

        return self.nn(out)

    def message(self, x_j, edge_attr=None):
        if edge_attr is None:
            return ops.relu(x_j)
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        return ops.relu(x_j + edge_attr)
