from typing import Optional, Tuple
from keras import ops
import numpy as np


def pool_edge(
    cluster,
    edge_index,
    edge_attr: Optional[any] = None,
    reduce: str = "sum",
) -> Tuple[any, Optional[any]]:
    r"""Pools edge indices and attributes based on cluster assignments.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import pool_edge

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        cluster = np.repeat(np.arange(5), 2)  # merge nodes pairwise into 5 clusters

        edge_attr = np.random.rand(30, 3).astype("float32")
        edge_index_pool, edge_attr_pool = pool_edge(cluster, edge_index, edge_attr)  # coarsened, deduplicated edges
        print(edge_index_pool.shape[0], edge_attr_pool.shape[1])  # 2 3
        ```
    """
    cluster_np = ops.convert_to_numpy(cluster)
    edge_index_np = ops.convert_to_numpy(edge_index)

    row = cluster_np[edge_index_np[0]]
    col = cluster_np[edge_index_np[1]]

    # Remove self-loops
    non_loop = row != col
    row = row[non_loop]
    col = col[non_loop]

    if len(row) == 0:
        empty_ei = ops.zeros((2, 0), dtype=edge_index.dtype)
        empty_ea = None if edge_attr is None else ops.zeros((0, *ops.shape(edge_attr)[1:]), dtype=edge_attr.dtype)
        return empty_ei, empty_ea

    edges = np.stack([row, col], axis=0)
    # Coalesce duplicate edges
    unique_edges, inv = np.unique(edges, axis=1, return_inverse=True)

    out_edge_index = ops.convert_to_tensor(unique_edges, dtype=edge_index.dtype)

    out_edge_attr = None
    if edge_attr is not None:
        ea_np = ops.convert_to_numpy(edge_attr)[non_loop]
        num_unique = unique_edges.shape[1]
        ea_tensor = ops.convert_to_tensor(ea_np, dtype=edge_attr.dtype)
        inv_tensor = ops.convert_to_tensor(inv, dtype="int32")
        if reduce == "sum":
            out_edge_attr = ops.segment_sum(ea_tensor, inv_tensor, num_segments=num_unique)
        elif reduce == "mean":
            sum_ea = ops.segment_sum(ea_tensor, inv_tensor, num_segments=num_unique)
            count = ops.segment_sum(ops.ones_like(ea_tensor), inv_tensor, num_segments=num_unique)
            out_edge_attr = sum_ea / ops.maximum(count, 1.0)
        elif reduce == "max":
            out_edge_attr = ops.segment_max(ea_tensor, inv_tensor, num_segments=num_unique)

    return out_edge_index, out_edge_attr


def pool_batch(perm, batch):
    r"""Pools batch vector given representative indices `perm`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import pool_batch

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        batch = np.repeat([0, 1], 5)  # nodes 0-4 belong to graph 0, nodes 5-9 to graph 1

        perm = np.array([0, 3, 6, 9])  # nodes kept after pooling
        print(tuple(pool_batch(perm, batch).shape))  # (4,): graph id of each kept node
        ```
    """
    return ops.take(batch, perm, axis=0)


def pool_pos(cluster, pos):
    r"""Pools node positions by computing average coordinates within each cluster.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import pool_pos

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        pos = np.random.rand(10, 3).astype("float32")  # 3D positions
        cluster = np.repeat(np.arange(5), 2)  # merge nodes pairwise into 5 clusters

        print(tuple(pool_pos(cluster, pos).shape))  # (5, 3): mean position of every cluster
        ```
    """
    cluster = ops.cast(cluster, dtype="int32")
    num_clusters = int(ops.max(cluster)) + 1 if ops.shape(cluster)[0] > 0 else 0
    sum_pos = ops.segment_sum(pos, cluster, num_segments=num_clusters)
    count = ops.segment_sum(ops.ones_like(pos), cluster, num_segments=num_clusters)
    return sum_pos / ops.maximum(count, 1.0)
