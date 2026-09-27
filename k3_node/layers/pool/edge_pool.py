from typing import Callable, List, NamedTuple, Optional, Tuple
from keras import layers, ops
import numpy as np


class UnpoolInfo(NamedTuple):
    edge_index: any
    cluster: any
    batch: any
    new_edge_score: any


class EdgePooling(layers.Layer):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`__ papers.
    """
    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: float = 0.0,
        add_to_edge_score: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout_rate = dropout
        self.drop = layers.Dropout(dropout) if dropout > 0.0 else None

        self.lin = layers.Dense(1, use_bias=True, name="lin")

    def build(self, input_shape=None):
        self.lin.build((None, 2 * self.in_channels))
        super().build(input_shape)

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes: int):
        col = ops.cast(edge_index[1], dtype="int32")
        max_score = ops.segment_max(raw_edge_score, col, num_segments=num_nodes)
        max_exp = ops.take(max_score, col, axis=0)
        exp_score = ops.exp(raw_edge_score - max_exp)
        sum_exp = ops.segment_sum(exp_score, col, num_segments=num_nodes)
        sum_exp_taken = ops.take(sum_exp, col, axis=0)
        return exp_score / (sum_exp_taken + 1e-12)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index=None, num_nodes=None):
        return ops.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index=None, num_nodes=None):
        return ops.sigmoid(raw_edge_score)

    def call(
        self,
        x,
        edge_index,
        batch,
        training: bool = False,
    ) -> Tuple[any, any, any, UnpoolInfo]:
        r"""Forward pass."""
        row = ops.cast(edge_index[0], dtype="int32")
        col = ops.cast(edge_index[1], dtype="int32")

        e = ops.concatenate([ops.take(x, row, axis=0), ops.take(x, col, axis=0)], axis=-1)
        e = ops.reshape(self.lin(e), (-1,))
        if self.drop is not None:
            e = self.drop(e, training=training)

        num_nodes = ops.shape(x)[0]
        e = self.compute_edge_score(e, edge_index, num_nodes)
        e = e + self.add_to_edge_score

        return self._merge_edges(x, edge_index, batch, e)

    def _merge_edges(
        self,
        x,
        edge_index,
        batch,
        edge_score,
    ) -> Tuple[any, any, any, UnpoolInfo]:
        from k3_node.layers.conv.utils import eager_only_placeholder, host_callback
        if eager_only_placeholder("EdgePooling", x, edge_index, edge_score):
            num_nodes = ops.shape(x)[0]
            unpool_info = UnpoolInfo(edge_index, ops.arange(num_nodes, dtype="int32"), batch, ops.ones((num_nodes,), dtype=x.dtype))
            return x, edge_index, batch, unpool_info

        num_nodes = int(ops.shape(x)[0])
        num_edges = int(ops.shape(edge_index)[1])

        def contract(edge_index_np, edge_score_np, batch_np):
            # Greedy contraction on the host; outputs are padded to fixed sizes (N clusters, E edges).
            cluster_np = np.zeros(num_nodes, dtype=np.int64)
            cluster_edge = np.full(num_nodes, -1, dtype=np.int64)  # edge merged into each cluster
            mask = np.ones(num_nodes, dtype=bool)
            i = 0
            for edge_idx in np.argsort(-edge_score_np, kind="stable"):
                source, target = edge_index_np[0, edge_idx], edge_index_np[1, edge_idx]
                if not mask[source] or not mask[target]:
                    continue
                cluster_edge[i] = edge_idx
                cluster_np[source] = i
                mask[source] = False
                if source != target:
                    cluster_np[target] = i
                    mask[target] = False
                i += 1
            remaining = np.where(mask)[0]
            cluster_np[remaining] = np.arange(i, i + len(remaining))
            num_clusters = i + len(remaining)

            unique_edges = np.unique(cluster_np[edge_index_np.astype(np.int64)], axis=1)
            edges_pad = np.zeros((2, num_edges), dtype=np.int64)
            edges_pad[:, : unique_edges.shape[1]] = unique_edges

            batch_pad = np.zeros(num_nodes, dtype=np.int64)
            batch_pad[cluster_np] = batch_np
            return cluster_np, cluster_edge, num_clusters, edges_pad, unique_edges.shape[1], batch_pad

        cluster, cluster_edge, num_clusters, edges_pad, num_new_edges, batch_pad = host_callback(
            contract,
            [((num_nodes,), "int32"), ((num_nodes,), "int32"), ((), "int32"),
             ((2, num_edges), "int32"), ((), "int32"), ((num_nodes,), "int32")],
            edge_index, edge_score, batch,
        )
        num_clusters, num_new_edges = int(num_clusters), int(num_new_edges)

        new_x = ops.segment_sum(x, cluster, num_segments=num_clusters)

        # Score of the edge merged into each cluster (1 for unmatched nodes), gathered from the score
        # tensor so gradients reach the scoring layer.
        cluster_edge = cluster_edge[:num_clusters]
        if num_edges > 0:
            gathered = ops.take(ops.cast(edge_score, x.dtype), ops.maximum(cluster_edge, 0), axis=0)
            new_edge_score = ops.where(cluster_edge >= 0, gathered, ops.ones_like(gathered))
        else:
            new_edge_score = ops.ones((num_clusters,), dtype=x.dtype)
        new_x = new_x * ops.reshape(new_edge_score, (-1, 1))

        new_edge_index = ops.cast(edges_pad[:, :num_new_edges], edge_index.dtype)
        new_batch = ops.cast(batch_pad[:num_clusters], batch.dtype)

        unpool_info = UnpoolInfo(
            edge_index=edge_index,
            cluster=cluster,
            batch=batch,
            new_edge_score=new_edge_score,
        )

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(
        self,
        x,
        unpool_info: UnpoolInfo,
    ) -> Tuple[any, any, any]:
        r"""Unpools a previous edge pooling step."""
        new_x = x / ops.reshape(unpool_info.new_edge_score, (-1, 1))
        new_x = ops.take(new_x, ops.cast(unpool_info.cluster, dtype="int32"), axis=0)
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels})"
