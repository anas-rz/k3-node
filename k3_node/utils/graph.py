import numpy as np
from keras import ops
from typing import Optional, Tuple, Union


def edge_index_to_adjacency_matrix(edge_index):
    edge_index = ops.convert_to_tensor(edge_index)
    num_nodes = ops.max(edge_index) + 1
    adjacency_matrix = ops.zeros((num_nodes, num_nodes), dtype="float32")

    indices = ops.transpose(edge_index, axes=[1, 0])
    updates = ops.ones(shape=(ops.shape(edge_index)[1],), dtype="float32")
    adjacency_matrix = ops.scatter_update(adjacency_matrix, indices, updates)

    return adjacency_matrix


def has_self_loops(edge_index) -> bool:
    """Returns True if the graph contains self-loops."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    if edge_index_np.size == 0:
        return False
    return bool(np.any(edge_index_np[0] == edge_index_np[1]))


def contains_isolated_nodes(edge_index, num_nodes: Optional[int] = None) -> bool:
    """Returns True if the graph contains isolated nodes."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    if num_nodes is None:
        if edge_index_np.size == 0:
            return False
        num_nodes = int(np.max(edge_index_np)) + 1

    if num_nodes == 0:
        return False
    if edge_index_np.size == 0:
        return num_nodes > 0

    unique_nodes = np.unique(edge_index_np)
    return len(unique_nodes) < num_nodes


def is_undirected(edge_index, edge_attr=None, num_nodes: Optional[int] = None) -> bool:
    """Returns True if the graph is undirected."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    if edge_index_np.shape[1] == 0:
        return True

    row, col = edge_index_np[0], edge_index_np[1]
    # Check if edges (row, col) match (col, row)
    order1 = np.lexsort((col, row))
    order2 = np.lexsort((row, col))

    if not np.array_equal(row[order1], col[order2]) or not np.array_equal(col[order1], row[order2]):
        return False

    if edge_attr is not None:
        attr_np = ops.convert_to_numpy(edge_attr)
        if not np.allclose(attr_np[order1], attr_np[order2], atol=1e-5):
            return False

    return True


def coalesce(
    edge_index,
    edge_attr=None,
    num_nodes: Optional[int] = None,
    is_sorted: bool = False,
    sort_by_row: bool = True,
    reduce: str = "add",
):
    """Sorts edge_index and removes duplicate edges, summing duplicate edge attributes."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    if edge_index_np.shape[1] == 0:
        return edge_index, edge_attr

    row, col = edge_index_np[0], edge_index_np[1]
    if sort_by_row:
        order = np.lexsort((col, row))
    else:
        order = np.lexsort((row, col))

    row = row[order]
    col = col[order]
    sorted_edge_index = np.stack([row, col], axis=0)

    # Find duplicates
    mask = np.ones(row.shape[0], dtype=bool)
    mask[1:] = (row[1:] != row[:-1]) | (col[1:] != col[:-1])

    if edge_attr is None:
        unique_edges = sorted_edge_index[:, mask]
        return ops.convert_to_tensor(unique_edges, dtype=edge_index.dtype), None

    attr_np = ops.convert_to_numpy(edge_attr)[order]
    if np.all(mask):
        return (
            ops.convert_to_tensor(sorted_edge_index, dtype=edge_index.dtype),
            ops.convert_to_tensor(attr_np, dtype=edge_attr.dtype),
        )

    # Accumulate attributes for duplicate edges
    unique_edges = sorted_edge_index[:, mask]
    group_idx = np.cumsum(mask) - 1
    num_unique = unique_edges.shape[1]

    if reduce == "add":
        out_attr = np.zeros((num_unique,) + attr_np.shape[1:], dtype=attr_np.dtype)
        np.add.at(out_attr, group_idx, attr_np)
    elif reduce == "mean":
        out_attr = np.zeros((num_unique,) + attr_np.shape[1:], dtype=attr_np.dtype)
        counts = np.zeros((num_unique,) + (1,) * (attr_np.ndim - 1), dtype=np.float32)
        np.add.at(out_attr, group_idx, attr_np)
        np.add.at(counts, group_idx, 1.0)
        out_attr = out_attr / np.maximum(counts, 1.0)
    else:
        out_attr = attr_np[mask]

    return (
        ops.convert_to_tensor(unique_edges, dtype=edge_index.dtype),
        ops.convert_to_tensor(out_attr, dtype=edge_attr.dtype),
    )


def subgraph(
    subset,
    edge_index,
    edge_attr=None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
):
    """Returns the induced subgraph of nodes in subset."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    subset_np = ops.convert_to_numpy(subset)

    if num_nodes is None:
        num_nodes = int(np.max(edge_index_np)) + 1 if edge_index_np.size > 0 else 0

    if subset_np.dtype == bool:
        node_mask = subset_np
        if node_mask.shape[0] < num_nodes:
            padded = np.zeros(num_nodes, dtype=bool)
            padded[: node_mask.shape[0]] = node_mask
            node_mask = padded
    else:
        node_mask = np.zeros(num_nodes, dtype=bool)
        node_mask[subset_np] = True

    edge_mask = node_mask[edge_index_np[0]] & node_mask[edge_index_np[1]]
    sub_edge_index = edge_index_np[:, edge_mask]

    sub_edge_attr = None
    if edge_attr is not None:
        attr_np = ops.convert_to_numpy(edge_attr)
        sub_edge_attr = attr_np[edge_mask]
        sub_edge_attr = ops.convert_to_tensor(sub_edge_attr, dtype=edge_attr.dtype)

    if relabel_nodes:
        node_idx = np.full(num_nodes, -1, dtype=edge_index_np.dtype)
        if subset_np.dtype == bool:
            node_idx[subset_np] = np.arange(np.sum(subset_np), dtype=edge_index_np.dtype)
        else:
            node_idx[subset_np] = np.arange(len(subset_np), dtype=edge_index_np.dtype)
        sub_edge_index = node_idx[sub_edge_index]

    sub_edge_index = ops.convert_to_tensor(sub_edge_index, dtype=edge_index.dtype)

    if return_edge_mask:
        edge_mask_tensor = ops.convert_to_tensor(edge_mask, dtype="bool")
        return sub_edge_index, sub_edge_attr, edge_mask_tensor
    return sub_edge_index, sub_edge_attr


def normalized_cut(edge_index, edge_attr, num_nodes: Optional[int] = None):
    r"""Computes the normalized cut :math:`\mathbf{e}_{i,j} \cdot
    \left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)` of a weighted graph.

    Example:
        ```python
        import numpy as np
        from k3_node.utils import normalized_cut

        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])  # a path 0 - 1 - 2
        edge_attr = np.ones(4, dtype="float32")  # every edge has weight 1

        print(np.asarray(normalized_cut(edge_index, edge_attr, num_nodes=3)))  # [1.5 1.5 1.5 1.5]
        ```
    """
    from k3_node.ops.segment import segment_sum

    edge_index = ops.convert_to_tensor(edge_index)
    edge_attr = ops.convert_to_tensor(edge_attr)
    row, col = edge_index[0], edge_index[1]
    if num_nodes is None:
        num_nodes = int(ops.max(edge_index)) + 1
    deg_inv = 1.0 / segment_sum(ops.ones_like(edge_attr), row, num_segments=num_nodes)
    return edge_attr * (ops.take(deg_inv, row, axis=0) + ops.take(deg_inv, col, axis=0))


def k_hop_subgraph(node_idx, num_hops: int, edge_index, relabel_nodes: bool = False,
                   num_nodes: Optional[int] = None, flow: str = "source_to_target"):
    r"""Finds the nodes within ``num_hops`` hops of ``node_idx`` and the edges among them.

    Returns ``(subset, edge_index, mapping, edge_mask)`` as in PyG: the nodes involved, the edges of
    the subgraph (relabeled to ``0..len(subset)-1`` if ``relabel_nodes``), the positions of
    ``node_idx`` in ``subset``, and which of the original edges were kept.

    Example:
        ```python
        import numpy as np
        from k3_node.utils import k_hop_subgraph

        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])  # a path 0 -> 1 -> 2 -> 3 -> 4
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(3, 2, edge_index, relabel_nodes=True)
        print(subset, sub_edge_index.tolist())  # [1 2 3] [[0, 1], [1, 2]]
        ```
    """
    edge_index = np.asarray(ops.convert_to_numpy(edge_index)).astype(np.int64)
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.size else 0
    row, col = (edge_index[0], edge_index[1]) if flow == "source_to_target" else (edge_index[1], edge_index[0])
    node_idx = np.atleast_1d(np.asarray(ops.convert_to_numpy(node_idx)).astype(np.int64))

    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask = np.zeros(num_nodes, dtype=bool)
        node_mask[subsets[-1]] = True
        subsets.append(row[node_mask[col]])
    subset, inverse = np.unique(np.concatenate(subsets), return_inverse=True)
    mapping = inverse[: node_idx.shape[0]]

    node_mask = np.zeros(num_nodes, dtype=bool)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    sub_edge_index = edge_index[:, edge_mask]
    if relabel_nodes:
        new_id = np.full(num_nodes, -1, dtype=np.int64)
        new_id[subset] = np.arange(subset.shape[0])
        sub_edge_index = new_id[sub_edge_index]
    return subset, sub_edge_index, mapping, edge_mask


def drnl_node_labeling(edge_index, src: int, dst: int, num_nodes: int):
    r"""Double-radius node labeling from the SEAL paper: labels every node of an enclosing
    subgraph by its distances to the two target nodes ``src`` and ``dst`` (which get label 1;
    unreachable nodes get 0).

    Example:
        ```python
        import numpy as np
        from k3_node.utils import drnl_node_labeling

        edge_index = np.array([[0, 1, 2, 2, 3, 1], [2, 2, 0, 1, 1, 3]])  # 0-2, 1-2, 1-3 in both directions
        print(drnl_node_labeling(edge_index, src=0, dst=1, num_nodes=4))  # [1 1 2 0]
        ```
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    edge_index = np.asarray(ops.convert_to_numpy(edge_index)).astype(np.int64)
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tocsr()
    if src == dst:  # a self-loop link: both distances are the same
        d = shortest_path(adj, directed=False, unweighted=True, indices=src)
        with np.errstate(invalid="ignore"):
            z = 1 + d * d
        z[src] = 1.0
        z[np.isnan(z) | np.isinf(z)] = 0.0
        return z.astype(np.int64)
    keep = [i for i in range(num_nodes) if i != src]
    adj_wo_src = adj[keep, :][:, keep]
    keep = [i for i in range(num_nodes) if i != dst]
    adj_wo_dst = adj[keep, :][:, keep]

    dist2src = np.insert(shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src), dst, 0, axis=0)
    dist2dst = np.insert(shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1), src, 0, axis=0)
    with np.errstate(invalid="ignore"):  # unreachable nodes have infinite distance
        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2
        z = 1 + np.minimum(dist2src, dist2dst) + dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = z[dst] = 1.0
    z[np.isnan(z) | np.isinf(z)] = 0.0
    return z.astype(np.int64)
