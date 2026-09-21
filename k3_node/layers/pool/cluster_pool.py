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
        from k3_node.layers.conv.utils import is_tracing
        if is_tracing(x) or is_tracing(edge_index):
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

        edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)
        edge_score_np = ops.convert_to_numpy(edge_score)
        num_nodes = ops.shape(x)[0]

        contract_mask = edge_score_np > self.threshold
        edge_contract = edge_index_np[:, contract_mask]

        if edge_contract.shape[1] > 0:
            adj = coo_matrix(
                (np.ones(edge_contract.shape[1]), (edge_contract[0], edge_contract[1])),
                shape=(num_nodes, num_nodes),
            )
            _, cluster_np = connected_components(adj, directed=True, connection="weak")
        else:
            cluster_np = np.arange(num_nodes)

        num_clusters = int(np.max(cluster_np)) + 1
        cluster = ops.convert_to_tensor(cluster_np, dtype="int32")

        # Dense assignment matrix C of shape [N, num_clusters]
        C_np = np.zeros((num_nodes, num_clusters), dtype=np.float32)
        C_np[np.arange(num_nodes), cluster_np] = 1.0

        # Dense adjacency A
        A_np = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if edge_index_np.shape[1] > 0:
            A_np[edge_index_np[0], edge_index_np[1]] = 1.0

        # Dense edge score S
        S_np = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if edge_index_np.shape[1] > 0:
            S_np[edge_index_np[0], edge_index_np[1]] = edge_score_np

        # Single nodes in contract graph
        A_contract = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if edge_contract.shape[1] > 0:
            A_contract[edge_contract[0], edge_contract[1]] = 1.0
        deg_contract = A_contract.sum(axis=-1) + A_contract.sum(axis=-2)
        nodes_single = np.where(deg_contract == 0)[0]
        S_np[nodes_single, nodes_single] = 1.0

        C = ops.convert_to_tensor(C_np, dtype=x.dtype)
        S = ops.convert_to_tensor(S_np, dtype=x.dtype)

        # x_out = (S @ C).t() @ x
        x_out = ops.matmul(ops.transpose(ops.matmul(S, C)), x)

        # Coarsened adjacency: (C.T @ A @ C) with zero diagonal
        coarse_A = np.dot(np.dot(C_np.T, A_np), C_np)
        np.fill_diagonal(coarse_A, 0.0)
        coarse_edges = np.where(coarse_A > 0)
        edge_index_out = ops.convert_to_tensor(
            np.stack([coarse_edges[0], coarse_edges[1]], axis=0),
            dtype=edge_index.dtype,
        )

        batch_np = ops.convert_to_numpy(batch)
        batch_out_np = np.empty(num_clusters, dtype=batch_np.dtype)
        batch_out_np[cluster_np] = batch_np
        batch_out = ops.convert_to_tensor(batch_out_np, dtype=batch.dtype)

        unpool_info = UnpoolInfo(edge_index, cluster, batch)
        return x_out, edge_index_out, batch_out, unpool_info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels})"
