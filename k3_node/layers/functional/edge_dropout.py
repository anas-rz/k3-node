import keras
from keras import ops


class EdgeDropout(keras.layers.Layer):
    r"""Randomly drops edges during training (DropEdge, `Rong et al., 2020
    <https://arxiv.org/abs/1907.10903>`_), like PyG's ``dropout_edge``.

    Instead of removing edges, which would change the number of edges in every step and cannot be
    compiled (XLA / ``jax.jit``), it returns an edge weight of 0 for dropped edges and 1 for kept
    ones. Pass it as ``edge_weight`` to layers that support edge weights (e.g. ``GCNConv``).

    Args:
        rate (float): Probability of dropping an edge. (default: ``0.5``)
        force_undirected (bool): Drop both directions ``(i, j)`` and ``(j, i)`` of an edge together.
            (default: :obj:`False`)

    Example:
        ```python
        import numpy as np
        from keras import ops
        from k3_node.layers import EdgeDropout

        edge_index = np.random.randint(0, 10, size=(2, 30))
        drop = EdgeDropout(rate=0.2, force_undirected=True)
        edge_weight = drop(edge_index, training=True)  # 0 for dropped edges, 1 for kept edges
        print(tuple(edge_weight.shape))  # (30,)
        print(float(ops.min(drop(edge_index))))  # 1.0: nothing is dropped at inference
        ```
    """

    def __init__(self, rate: float = 0.5, force_undirected: bool = False, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.force_undirected = force_undirected
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, edge_index, training=None):
        edge_index = ops.cast(edge_index, "int32")
        num_edges = ops.shape(edge_index)[1]
        if not training or self.rate <= 0:
            return ops.ones((num_edges,), dtype="float32")
        if not self.force_undirected:
            noise = keras.random.uniform((num_edges,), seed=self.seed_generator)
        else:
            # The same pseudo-random number for (i, j) and (j, i): hash the unordered node pair.
            lo = ops.cast(ops.minimum(edge_index[0], edge_index[1]), "float32")
            hi = ops.cast(ops.maximum(edge_index[0], edge_index[1]), "float32")
            offset = keras.random.uniform((), seed=self.seed_generator) * 1000.0
            h = ops.sin(lo * 12.9898 + hi * 78.233 + offset) * 43758.5453
            noise = h - ops.floor(h)
        return ops.cast(noise >= self.rate, "float32")

    def get_config(self):
        return {**super().get_config(), "rate": self.rate, "force_undirected": self.force_undirected}
