from typing import Optional, Tuple, Union
from keras import ops


def degree(index, num_nodes: Optional[int] = None, dtype=None):
    """Computes the (in/out) degree of a given index tensor.

    Args:
        index: 1D tensor of node indices.
        num_nodes: The number of nodes.
        dtype: Output data type.
    """
    index = ops.cast(index, "int32")
    if num_nodes is None:
        num_nodes = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0
    num_nodes = int(num_nodes)
    deg = ops.bincount(index, minlength=num_nodes)
    if dtype is not None:
        deg = ops.cast(deg, dtype)
    return deg


def remove_self_loops(
    edge_index,
    edge_attr=None,
) -> Tuple:
    """Removes self-loops from `edge_index` and optional `edge_attr`."""
    edge_index = ops.convert_to_tensor(edge_index)
    if edge_attr is not None:
        edge_attr = ops.convert_to_tensor(edge_attr)
    mask = edge_index[0] != edge_index[1]
    where_mask = ops.where(mask)
    indices = where_mask[0] if isinstance(where_mask, (list, tuple)) else where_mask
    indices = ops.reshape(indices, (-1,))
    edge_index = ops.take(edge_index, indices, axis=1)
    if edge_attr is not None:
        edge_attr = ops.take(edge_attr, indices, axis=0)
    return edge_index, edge_attr


def add_self_loops(
    edge_index,
    edge_attr=None,
    fill_value: Union[float, str, None] = None,
    num_nodes: Optional[int] = None,
) -> Tuple:
    """Adds self-loops to `edge_index` and optional `edge_attr`."""
    edge_index = ops.convert_to_tensor(edge_index)
    if num_nodes is None:
        num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0
    num_nodes = int(num_nodes)

    loop_index = ops.arange(0, num_nodes, dtype=edge_index.dtype)
    loop_index = ops.stack([loop_index, loop_index], axis=0)
    edge_index = ops.concatenate([edge_index, loop_index], axis=1)

    if edge_attr is not None:
        edge_attr = ops.convert_to_tensor(edge_attr)
        attr_shape = (num_nodes,) + tuple(ops.shape(edge_attr)[1:])
        if fill_value is None:
            loop_attr = ops.zeros(attr_shape, dtype=edge_attr.dtype)
        elif isinstance(fill_value, (int, float)):
            loop_attr = ops.full(attr_shape, fill_value, dtype=edge_attr.dtype)
        elif fill_value == "add" or fill_value == "mean":
            loop_attr = ops.zeros(attr_shape, dtype=edge_attr.dtype)
        else:
            loop_attr = ops.full(attr_shape, fill_value, dtype=edge_attr.dtype)
        edge_attr = ops.concatenate([edge_attr, loop_attr], axis=0)

    return edge_index, edge_attr


def gcn_norm(
    edge_index,
    edge_weight=None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype=None,
) -> Tuple:
    """Computes the GCN normalization coefficients."""
    fill_value = 2.0 if improved else 1.0
    edge_index = ops.convert_to_tensor(edge_index)
    if edge_weight is not None:
        edge_weight = ops.convert_to_tensor(edge_weight)

    if num_nodes is None:
        num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0
    num_nodes = int(num_nodes)

    if edge_weight is None:
        edge_weight = ops.ones((ops.shape(edge_index)[1],), dtype=dtype or "float32")

    if add_self_loops:
        edge_index, edge_weight = globals()["add_self_loops"](
            edge_index, edge_weight, fill_value=fill_value, num_nodes=num_nodes
        )

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    row_cast = ops.cast(row, "int32")
    col_cast = ops.cast(col, "int32")
    idx_cast = ops.cast(idx, "int32")

    deg = ops.bincount(idx_cast, weights=edge_weight, minlength=num_nodes)
    deg_inv_sqrt = ops.power(deg, -0.5)
    deg_inv_sqrt = ops.where(
        ops.isinf(deg_inv_sqrt) | ops.isnan(deg_inv_sqrt), 0.0, deg_inv_sqrt
    )

    norm = ops.take(deg_inv_sqrt, row_cast, axis=0) * edge_weight * ops.take(deg_inv_sqrt, col_cast, axis=0)
    return edge_index, norm


def get_laplacian(
    edge_index,
    edge_weight=None,
    normalization: Optional[str] = None,
    dtype=None,
    num_nodes: Optional[int] = None,
) -> Tuple:
    """Computes the graph Laplacian of the given graph."""
    edge_index = ops.convert_to_tensor(edge_index)
    if edge_weight is not None:
        edge_weight = ops.convert_to_tensor(edge_weight)

    if num_nodes is None:
        num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0
    num_nodes = int(num_nodes)

    if edge_weight is None:
        edge_weight = ops.ones((ops.shape(edge_index)[1],), dtype=dtype or "float32")

    row, col = ops.cast(edge_index[0], "int32"), ops.cast(edge_index[1], "int32")
    deg = degree(row, num_nodes=num_nodes, dtype=edge_weight.dtype)

    if normalization is None:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = ops.concatenate([-edge_weight, deg], axis=0)
    elif normalization == "sym":
        deg_inv_sqrt = ops.power(deg, -0.5)
        deg_inv_sqrt = ops.where(
            ops.isinf(deg_inv_sqrt) | ops.isnan(deg_inv_sqrt), 0.0, deg_inv_sqrt
        )
        edge_weight = (
            ops.take(deg_inv_sqrt, row, axis=0)
            * (-edge_weight)
            * ops.take(deg_inv_sqrt, col, axis=0)
        )
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = ops.concatenate(
            [edge_weight, ops.ones((num_nodes,), dtype=edge_weight.dtype)], axis=0
        )
    elif normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv = ops.where(
            ops.isinf(deg_inv) | ops.isnan(deg_inv), 0.0, deg_inv
        )
        edge_weight = ops.take(deg_inv, row, axis=0) * (-edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = ops.concatenate(
            [edge_weight, ops.ones((num_nodes,), dtype=edge_weight.dtype)], axis=0
        )
    return edge_index, edge_weight


def softmax(src, index, num_nodes: Optional[int] = None, dim: int = -2):
    """Computes a sparsely evaluated softmax over index."""
    index = ops.cast(index, "int32")
    if num_nodes is None:
        num_nodes = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0
    num_nodes = int(num_nodes)

    max_val = ops.segment_max(src, index, num_segments=num_nodes)
    max_val = ops.take(max_val, index, axis=dim)
    exp = ops.exp(src - max_val)
    sum_val = ops.segment_sum(exp, index, num_segments=num_nodes)
    sum_val = ops.take(sum_val, index, axis=dim)
    return exp / (sum_val + 1e-12)


def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """Computes scatter / segment reduction."""
    index = ops.cast(index, "int32")
    if dim_size is None:
        dim_size = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0
    dim_size = int(dim_size)

    if reduce in ("add", "sum"):
        return ops.segment_sum(src, index, num_segments=dim_size)
    elif reduce == "mean":
        sum_val = ops.segment_sum(src, index, num_segments=dim_size)
        ones = ops.ones_like(src)
        count = ops.segment_sum(ones, index, num_segments=dim_size)
        count = ops.maximum(count, 1.0)
        return sum_val / count
    elif reduce == "max":
        return ops.segment_max(src, index, num_segments=dim_size)
    elif reduce == "min":
        return ops.segment_min(src, index, num_segments=dim_size)
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")



