from typing import Optional, Tuple

import keras
from keras import ops


class MetaLayer(keras.layers.Layer):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity). The input graph has node features :obj:`x`,
    edge features :obj:`edge_attr` as well as graph-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the modules :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    To allow for batch-wise graph processing, all callable functions take an
    additional argument :obj:`batch`, which determines the assignment of
    edges or nodes to their specific graphs.

    Args:
        edge_model (callable, optional): A callable which updates a graph's
            edge features based on its source and target node features, its
            current edge features and its global features.
            (default: :obj:`None`)
        node_model (callable, optional): A callable which updates a graph's
            node features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (callable, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features.
            (default: :obj:`None`)

    Example::

        from keras import layers
        from k3_node.models import MetaLayer
        from k3_node.layers.conv.utils import scatter

        class EdgeModel(layers.Layer):
            def __init__(self):
                super().__init__()
                self.mlp = layers.Dense(5)
            def call(self, src, dst, edge_attr, u, batch):
                out = ops.concatenate([src, dst, edge_attr, u[batch]], axis=1)
                return self.mlp(out)

        class NodeModel(layers.Layer):
            def __init__(self):
                super().__init__()
                self.mlp1 = layers.Dense(10)
                self.mlp2 = layers.Dense(10)
            def call(self, x, edge_index, edge_attr, u, batch):
                row, col = edge_index[0], edge_index[1]
                out = ops.concatenate([x[row], edge_attr], axis=1)
                out = scatter(self.mlp1(out), col, dim_size=ops.shape(x)[0])
                out = ops.concatenate([x, out, u[batch]], axis=1)
                return self.mlp2(out)

        class GlobalModel(layers.Layer):
            def __init__(self):
                super().__init__()
                self.mlp = layers.Dense(20)
            def call(self, x, edge_index, edge_attr, u, batch):
                out = ops.concatenate([u, scatter(x, batch)], axis=1)
                return self.mlp(out)

        op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
        x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
    """

    def __init__(
        self,
        edge_model=None,
        node_model=None,
        global_model=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def call(
        self,
        x,
        edge_index,
        edge_attr=None,
        u=None,
        batch=None,
    ) -> Tuple:
        r"""Forward pass.

        Args:
            x (Tensor): The node features of shape ``[N, F_x]``.
            edge_index (Tensor): The edge indices of shape ``[2, E]``.
            edge_attr (Tensor, optional): The edge features of shape
                ``[E, F_e]``. (default: :obj:`None`)
            u (Tensor, optional): The global graph features of shape
                ``[B, F_u]``. (default: :obj:`None`)
            batch (Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`.
                (default: :obj:`None`)
        """
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_batch = batch if batch is None else ops.take(batch, row, axis=0)
            edge_attr = self.edge_model(
                ops.take(x, row, axis=0),
                ops.take(x, col, axis=0),
                edge_attr,
                u,
                edge_batch,
            )

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')

