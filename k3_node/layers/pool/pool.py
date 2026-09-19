from typing import Optional, Tuple
from keras import ops
import numpy as np


def pool_edge(
    cluster,
    edge_index,
    edge_attr: Optional[any] = None,
    reduce: str = "sum",
) -> Tuple[any, Optional[any]]:
    r"""Pools edge indices and attributes based on cluster assignments."""
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
    r"""Pools batch vector given representative indices `perm`."""
    return ops.take(batch, perm, axis=0)


def pool_pos(cluster, pos):
    r"""Pools node positions by computing average coordinates within each cluster."""
    cluster = ops.cast(cluster, dtype="int32")
    num_clusters = int(ops.max(cluster)) + 1 if ops.shape(cluster)[0] > 0 else 0
    sum_pos = ops.segment_sum(pos, cluster, num_segments=num_clusters)
    count = ops.segment_sum(ops.ones_like(pos), cluster, num_segments=num_clusters)
    return sum_pos / ops.maximum(count, 1.0)
