"""Graph topology construction strategies from tabular data."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class KNNGraphBuilder:
    r"""Constructs a k-nearest-neighbors graph from a node feature matrix.

    Args:
        k: Number of nearest neighbors per node. (default: ``5``)
        metric: Distance metric (``"cosine"``, ``"euclidean"``, or ``"manhattan"``).
            (default: ``"cosine"``)
        loop: Whether to include self-loops. (default: ``False``)
        bidirectional: Whether to make the resulting graph undirected. (default: ``True``)
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "cosine",
        loop: bool = False,
        bidirectional: bool = True,
    ):
        self.k = k
        self.metric = metric.lower()
        self.loop = loop
        self.bidirectional = bidirectional

    def __call__(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        num_nodes = x.shape[0]
        if num_nodes == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

        actual_k = min(self.k if self.loop else self.k + 1, num_nodes)

        if self.metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            x_norm = x / norms
            sim_matrix = np.dot(x_norm, x_norm.T)
            # Higher similarity is closer
            dist_matrix = 1.0 - sim_matrix
        elif self.metric == "euclidean":
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        elif self.metric == "manhattan":
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_matrix = np.sum(np.abs(diff), axis=-1)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Supported: 'cosine', 'euclidean', 'manhattan'.")

        src_list = []
        dst_list = []
        dist_list = []

        for i in range(num_nodes):
            row_dists = dist_matrix[i]
            nearest_indices = np.argsort(row_dists)
            count = 0
            for neighbor in nearest_indices:
                if not self.loop and neighbor == i:
                    continue
                src_list.append(i)
                dst_list.append(neighbor)
                dist_list.append(row_dists[neighbor])
                count += 1
                if count >= self.k:
                    break

        if self.bidirectional:
            # Add reverse edges if not already present
            edge_set = set(zip(src_list, dst_list))
            for s, d, w in list(zip(src_list, dst_list, dist_list)):
                if (d, s) not in edge_set:
                    src_list.append(d)
                    dst_list.append(s)
                    dist_list.append(w)
                    edge_set.add((d, s))

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.array(dist_list, dtype=np.float32).reshape(-1, 1) if dist_list else np.empty((0, 1), dtype=np.float32)
        return edge_index, edge_attr


class SimilarityGraphBuilder:
    r"""Constructs a graph connecting node pairs whose pairwise similarity exceeds a threshold.

    Args:
        threshold: Minimum similarity required to create an edge. (default: ``0.7``)
        metric: Similarity function (``"cosine"`` or ``"rbf"``). (default: ``"cosine"``)
        gamma: Bandwidth parameter for RBF kernel. (default: ``1.0``)
        loop: Whether to include self-loops. (default: ``False``)
    """

    def __init__(
        self,
        threshold: float = 0.7,
        metric: str = "cosine",
        gamma: float = 1.0,
        loop: bool = False,
    ):
        self.threshold = threshold
        self.metric = metric.lower()
        self.gamma = gamma
        self.loop = loop

    def __call__(self, x: np.ndarray, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        num_nodes = x.shape[0]
        if num_nodes == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

        if self.metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            x_norm = x / norms
            sim_matrix = np.dot(x_norm, x_norm.T)
        elif self.metric == "rbf":
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            sim_matrix = np.exp(-self.gamma * sq_dist)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'. Supported: 'cosine', 'rbf'.")

        if not self.loop:
            np.fill_diagonal(sim_matrix, -np.inf)

        src, dst = np.where(sim_matrix >= self.threshold)
        weights = sim_matrix[src, dst].astype(np.float32).reshape(-1, 1)
        edge_index = np.stack([src, dst], axis=0).astype(np.int64)
        return edge_index, weights


class SharedEntityGraphBuilder:
    r"""Connects tabular rows that share one or more categorical identifier values.
    (e.g., users sharing the same IP address, device, category, or cluster).

    Args:
        entity_cols: List of column names to check for shared values.
        max_degree: Maximum number of neighbors created per shared entity (to avoid supernode explosion).
            (default: ``50``)
        loop: Whether to include self-loops. (default: ``False``)
    """

    def __init__(
        self,
        entity_cols: Sequence[str],
        max_degree: int = 50,
        loop: bool = False,
    ):
        self.entity_cols = list(entity_cols)
        self.max_degree = max_degree
        self.loop = loop

    def __call__(self, x: np.ndarray, df_or_dict: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if df_or_dict is None:
            raise ValueError("SharedEntityGraphBuilder requires 'df_or_dict' containing the entity columns.")

        num_nodes = x.shape[0]
        src_list = []
        dst_list = []

        from k3_node.etl.encoders import _get_column_values

        for col in self.entity_cols:
            vals = _get_column_values(df_or_dict, col)
            val_to_rows: Dict[Any, List[int]] = {}
            for row_idx, val in enumerate(vals):
                if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
                    continue
                val_to_rows.setdefault(val, []).append(row_idx)

            for val, rows in val_to_rows.items():
                if len(rows) > self.max_degree:
                    # Subsample if group is too large
                    sampled_rows = np.random.choice(rows, size=self.max_degree, replace=False).tolist()
                else:
                    sampled_rows = rows

                for i in sampled_rows:
                    for j in sampled_rows:
                        if not self.loop and i == j:
                            continue
                        src_list.append(i)
                        dst_list.append(j)

        if not src_list:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

        edges = list(set(zip(src_list, dst_list)))
        src_arr = np.array([e[0] for e in edges], dtype=np.int64)
        dst_arr = np.array([e[1] for e in edges], dtype=np.int64)
        edge_index = np.stack([src_arr, dst_arr], axis=0)
        edge_attr = np.ones((len(edges), 1), dtype=np.float32)
        return edge_index, edge_attr


class SequentialGraphBuilder:
    r"""Connects tabular rows sequentially in order of an index or timestamp column,
    optionally partitioned by a group entity column.

    Args:
        order_col: Optional column name used to sort rows (e.g. timestamp or sequence index).
        group_by_col: Optional column name to partition sequences (e.g. user_id or session_id).
        window_size: Number of forward/backward sequential steps to connect. (default: ``1``)
        bidirectional: Whether to create undirected edges. (default: ``True``)
    """

    def __init__(
        self,
        order_col: Optional[str] = None,
        group_by_col: Optional[str] = None,
        window_size: int = 1,
        bidirectional: bool = True,
    ):
        self.order_col = order_col
        self.group_by_col = group_by_col
        self.window_size = window_size
        self.bidirectional = bidirectional

    def __call__(self, x: np.ndarray, df_or_dict: Optional[Any] = None, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        num_nodes = x.shape[0]
        if num_nodes == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

        from k3_node.etl.encoders import _get_column_values

        if self.group_by_col is not None and df_or_dict is not None:
            groups = _get_column_values(df_or_dict, self.group_by_col)
            group_to_indices: Dict[Any, List[int]] = {}
            for idx, g in enumerate(groups):
                group_to_indices.setdefault(g, []).append(idx)
        else:
            group_to_indices = {"all": list(range(num_nodes))}

        if self.order_col is not None and df_or_dict is not None:
            order_vals = _get_column_values(df_or_dict, self.order_col)
        else:
            order_vals = None

        src_list = []
        dst_list = []

        for group_name, row_indices in group_to_indices.items():
            if order_vals is not None:
                sorted_indices = sorted(row_indices, key=lambda idx: order_vals[idx])
            else:
                sorted_indices = row_indices

            n_seq = len(sorted_indices)
            for i in range(n_seq):
                curr_node = sorted_indices[i]
                for step in range(1, self.window_size + 1):
                    if i + step < n_seq:
                        next_node = sorted_indices[i + step]
                        src_list.append(curr_node)
                        dst_list.append(next_node)
                        if self.bidirectional:
                            src_list.append(next_node)
                            dst_list.append(curr_node)

        if not src_list:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

        edges = list(set(zip(src_list, dst_list)))
        src_arr = np.array([e[0] for e in edges], dtype=np.int64)
        dst_arr = np.array([e[1] for e in edges], dtype=np.int64)
        edge_index = np.stack([src_arr, dst_arr], axis=0)
        edge_attr = np.ones((len(edges), 1), dtype=np.float32)
        return edge_index, edge_attr
