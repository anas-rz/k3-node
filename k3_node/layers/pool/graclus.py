from typing import Optional
from keras import ops
import numpy as np


def graclus(
    edge_index,
    weight: Optional[any] = None,
    num_nodes: Optional[int] = None,
):
    r"""A greedy clustering algorithm of picking an unmarked vertex and matching
    it with one of its unmarked neighbors that maximizes its edge weight.
    """
    edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)
    if num_nodes is None:
        num_nodes = int(np.max(edge_index_np)) + 1 if edge_index_np.size > 0 else 0

    if weight is not None:
        weight_np = ops.convert_to_numpy(weight)
    else:
        weight_np = np.ones(edge_index_np.shape[1], dtype=np.float32)

    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    for idx in range(edge_index_np.shape[1]):
        u = edge_index_np[0, idx]
        v = edge_index_np[1, idx]
        w = weight_np[idx]
        adj[u].append((v, w))

    cluster_np = np.full(num_nodes, -1, dtype=np.int64)
    c = 0
    for u in range(num_nodes):
        if cluster_np[u] != -1:
            continue

        best_v = None
        best_w = -float("inf")
        for v, w in adj[u]:
            if v != u and cluster_np[v] == -1:
                if w > best_w:
                    best_w = w
                    best_v = v

        if best_v is not None:
            cluster_np[u] = c
            cluster_np[best_v] = c
        else:
            cluster_np[u] = c
        c += 1

    return ops.convert_to_tensor(cluster_np, dtype="int64")

