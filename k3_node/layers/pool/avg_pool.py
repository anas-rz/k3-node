from typing import Callable, Optional, Tuple
from keras import ops

from .consecutive import consecutive_cluster
from .pool import pool_batch, pool_edge, pool_pos


def _avg_pool_x(cluster, x, size: Optional[int] = None):
    cluster = ops.cast(cluster, dtype="int32")
    if size is None:
        size = int(ops.max(cluster)) + 1 if ops.shape(cluster)[0] > 0 else 0
    sum_x = ops.segment_sum(x, cluster, num_segments=size)
    ones = ops.ones_like(x)
    count = ops.segment_sum(ones, cluster, num_segments=size)
    return sum_x / ops.maximum(count, 1.0)


def avg_pool_x(
    cluster,
    x,
    batch,
    batch_size: Optional[int] = None,
    size: Optional[int] = None,
) -> Tuple[any, Optional[any]]:
    r"""Average-pools node features according to the clustering defined in `cluster`."""
    if size is not None:
        if batch_size is None:
            batch_size = int(ops.max(batch)) + 1
        return _avg_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _avg_pool_x(cluster, x)
    batch = pool_batch(perm, batch)
    return x, batch


def avg_pool(
    cluster,
    data,
    transform: Optional[Callable] = None,
    edge_index: Optional[any] = None,
    edge_attr: Optional[any] = None,
    batch: Optional[any] = None,
    pos: Optional[any] = None,
):
    r"""Pools and coarsens a graph given by `data` according to `cluster` using averaging."""
    cluster, perm = consecutive_cluster(cluster)

    if hasattr(data, "x"):
        x = getattr(data, "x", None)
        if x is not None:
            data.x = _avg_pool_x(cluster, x)

        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_index is not None:
            data.edge_index, data.edge_attr = pool_edge(cluster, edge_index, edge_attr, reduce="mean")

        batch = getattr(data, "batch", None)
        if batch is not None:
            data.batch = pool_batch(perm, batch)

        pos = getattr(data, "pos", None)
        if pos is not None:
            data.pos = pool_pos(cluster, pos)

        if transform is not None:
            data = transform(data)

        return data

    # Raw tensor mode
    pooled_x = _avg_pool_x(cluster, data)
    pooled_edge_index, pooled_edge_attr = (None, None)
    if edge_index is not None:
        pooled_edge_index, pooled_edge_attr = pool_edge(cluster, edge_index, edge_attr, reduce="mean")
    pooled_batch = pool_batch(perm, batch) if batch is not None else None

    return pooled_x, pooled_edge_index, pooled_batch


def avg_pool_neighbor_x(
    data,
    edge_index=None,
    flow: str = "source_to_target",
):
    r"""Average-pools neighboring node features."""
    if hasattr(data, "x"):
        x = data.x
        edge_index = data.edge_index
        is_data_obj = True
    else:
        x = data
        is_data_obj = False
        if edge_index is None:
            raise ValueError("edge_index must be provided if data is a tensor.")

    num_nodes = getattr(data, "num_nodes", None) if is_data_obj else ops.shape(x)[0]
    if num_nodes is None:
        num_nodes = ops.shape(x)[0]

    # Add self-loops
    loop_idx = ops.arange(num_nodes, dtype=edge_index.dtype)
    loop_edge = ops.stack([loop_idx, loop_idx], axis=0)
    full_edge_index = ops.concatenate([edge_index, loop_edge], axis=1)

    row = full_edge_index[0]
    col = full_edge_index[1]
    row, col = (row, col) if flow == "source_to_target" else (col, row)

    col = ops.cast(col, dtype="int32")
    x_src = ops.take(x, row, axis=0)
    sum_x = ops.segment_sum(x_src, col, num_segments=num_nodes)
    ones = ops.ones_like(x_src)
    count = ops.segment_sum(ones, col, num_segments=num_nodes)
    out_x = sum_x / ops.maximum(count, 1.0)
    if is_data_obj:
        data.x = out_x
        return data
    return out_x
