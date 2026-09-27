from typing import NamedTuple, Optional, Tuple
from keras import layers, ops
import numpy as np


class UnpoolInfo(NamedTuple):
    edge_index: any
    cluster: any
    batch: any


class ClusterPooling(layers.Layer):
    r"""The cluster pooling operator from the `"Edge-Based Graph Component
    Pooling" <https://arxiv.org/abs/2409.11856>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import ClusterPooling

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        batch = np.repeat([0, 1], 5)  # nodes 0-4 belong to graph 0, nodes 5-9 to graph 1

        layer = ClusterPooling(in_channels=8)
        x_pool, edge_index_pool, batch_pool, unpool_info = layer(x, edge_index, batch)
        # The number of clusters depends on the learned edge scores
        print(x_pool.shape[0] <= 10, x_pool.shape[1])  # True 8: fewer nodes, same features
        ```
    """
    def __init__(
        self,
        in_channels: int,
        edge_score_method: str = "tanh",
        dropout: float = 0.0,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert edge_score_method in ["tanh", "sigmoid", "log_softmax"]

        if threshold is None:
            threshold = 0.5 if edge_score_method == "sigmoid" else 0.0

        self.in_channels = in_channels
        self.edge_score_method = edge_score_method
        self.dropout_rate = dropout
        self.drop = layers.Dropout(dropout) if dropout > 0.0 else None
        self.threshold = threshold

        self.lin = layers.Dense(1, use_bias=True, name="lin")

    def build(self, input_shape=None):
        self.lin.build((None, 2 * self.in_channels))
        super().build(input_shape)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def call(
        self,
        x,
        edge_index,
        batch,
        training: bool = False,
    ) -> Tuple[any, any, any, UnpoolInfo]:
        r"""Forward pass."""
        from k3_node.layers.conv.utils import eager_only_placeholder
        if eager_only_placeholder("ClusterPooling", x, edge_index):
            num_nodes = ops.shape(x)[0]
            unpool_info = UnpoolInfo(edge_index, ops.arange(num_nodes, dtype="int32"), batch)
            return x, edge_index, batch, unpool_info

        edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)
        mask = edge_index_np[0] != edge_index_np[1]
        edge_index_filtered = edge_index_np[:, mask]

        row = ops.convert_to_tensor(edge_index_filtered[0], dtype="int32")
        col = ops.convert_to_tensor(edge_index_filtered[1], dtype="int32")

        edge_attr = ops.concatenate([ops.take(x, row, axis=0), ops.take(x, col, axis=0)], axis=-1)
        edge_score = ops.reshape(self.lin(edge_attr), (-1,))
        if self.drop is not None:
            edge_score = self.drop(edge_score, training=training)

        if self.edge_score_method == "tanh":
            edge_score = ops.tanh(edge_score)
        elif self.edge_score_method == "sigmoid":
            edge_score = ops.sigmoid(edge_score)
        else:
            edge_score = ops.log_softmax(edge_score, axis=0)

        edge_index_tensor = ops.convert_to_tensor(edge_index_filtered, dtype=edge_index.dtype)
        return self._merge_edges(x, edge_index_tensor, batch, edge_score)

    def _merge_edges(
        self,
        x,
        edge_index,
        batch,
        edge_score,
    ) -> Tuple[any, any, any, UnpoolInfo]:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        from k3_node.layers.conv.utils import host_callback

        num_nodes = int(ops.shape(x)[0])
        num_edges = int(ops.shape(edge_index)[1])
        threshold = self.threshold

        def contract(edge_index_np, edge_score_np, batch_np):
            # Clusters are the weakly connected components of the edges scoring above the threshold.
            edge_index_np = edge_index_np.astype(np.int64)
            edge_contract = edge_index_np[:, edge_score_np > threshold]
            if edge_contract.shape[1] > 0:
                adj = coo_matrix(
                    (np.ones(edge_contract.shape[1]), (edge_contract[0], edge_contract[1])),
                    shape=(num_nodes, num_nodes),
                )
                _, cluster_np = connected_components(adj, directed=True, connection="weak")
            else:
                cluster_np = np.arange(num_nodes)
            num_clusters = int(np.max(cluster_np)) + 1 if num_nodes > 0 else 0

            # Nodes without any contracted edge keep their own features (unit diagonal score).
            single = np.ones(num_nodes, dtype=bool)
            single[edge_contract[0]] = False
            single[edge_contract[1]] = False

            # Coarsened edges between distinct clusters, in (row, col) order.
            pairs = cluster_np[edge_index_np]
            pairs = np.unique(pairs[:, pairs[0] != pairs[1]], axis=1)
            edges_pad = np.zeros((2, num_edges), dtype=np.int64)
            edges_pad[:, : pairs.shape[1]] = pairs

            batch_pad = np.zeros(num_nodes, dtype=np.int64)
            batch_pad[cluster_np] = batch_np
            return cluster_np, num_clusters, single, edges_pad, pairs.shape[1], batch_pad

        cluster, num_clusters, single, edges_pad, num_new_edges, batch_pad = host_callback(
            contract,
            [((num_nodes,), "int32"), ((), "int32"), ((num_nodes,), "float32"),
             ((2, num_edges), "int32"), ((), "int32"), ((num_nodes,), "int32")],
            edge_index, edge_score, batch,
        )
        num_clusters, num_new_edges = int(num_clusters), int(num_new_edges)

        # x_out = (S @ C)^T @ x, computed sparsely: every edge (row -> col) adds score * x[row] to
        # cluster(col), and every unmatched node adds its own features to its cluster. The score
        # enters as a tensor so gradients reach the scoring layer.
        row = ops.cast(edge_index[0], "int32")
        col = ops.cast(edge_index[1], "int32")
        msgs = ops.expand_dims(ops.cast(edge_score, x.dtype), -1) * ops.take(x, row, axis=0)
        x_out = ops.segment_sum(msgs, ops.take(cluster, col, axis=0), num_segments=num_clusters)
        x_out = x_out + ops.segment_sum(
            x * ops.expand_dims(ops.cast(single, x.dtype), -1), cluster, num_segments=num_clusters
        )

        edge_index_out = ops.cast(edges_pad[:, :num_new_edges], edge_index.dtype)
        batch_out = ops.cast(batch_pad[:num_clusters], batch.dtype)

        unpool_info = UnpoolInfo(edge_index, cluster, batch)
        return x_out, edge_index_out, batch_out, unpool_info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels})"
