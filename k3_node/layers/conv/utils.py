from typing import Any, Optional, Tuple, Union
from keras import ops
from k3_node.ops.segment import segment_max, segment_min, segment_sum
from k3_node.ops.creation import full


def is_tracing(x: Any) -> bool:
    if x is None:
        return False
    try:
        from keras.src.backend.common.symbolic_scope import in_symbolic_scope

        if in_symbolic_scope():
            return True
    except Exception:
        pass
    name = type(x).__name__
    if "Tracer" in name or "KerasTensor" in name or "SymbolicTensor" in name:
        return True
    if hasattr(x, "_trace"):
        return True
    try:
        import jax
        if isinstance(x, jax.core.Tracer):
            return True
    except Exception:
        pass
    try:
        import tensorflow as tf
        if hasattr(x, "graph") and getattr(x, "graph", None) is not None:
            if not tf.executing_eagerly():
                return True
    except Exception:
        pass
    return False


def _is_compiled_trace(x) -> bool:
    """True inside compiled functions, where tensor values are unknown.

    On JAX, autodiff tracers created by ``jax.grad`` in eager mode (``run_eagerly=True``)
    still carry concrete values, so they do not count as compiled.
    """
    import keras

    if keras.config.backend() != "jax":
        return is_tracing(x)
    import jax
    import jax.numpy as jnp

    if not isinstance(x, jax.core.Tracer):
        return False
    try:
        int(jnp.sum(jnp.ravel(x)[:1] * 0))  # concretizing fails only inside jit
        return False
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerIntegerConversionError):
        return True


def host_callback(fn, out_specs, *args):
    """Runs the NumPy function ``fn`` on the concrete values of ``args``.

    ``out_specs`` is a sequence of ``(shape, dtype)`` giving the fixed shapes of ``fn``'s outputs.
    No gradient flows through the outputs. On JAX this uses ``jax.pure_callback``, which also
    works under ``jax.grad``; elsewhere the arguments are converted to NumPy directly.
    """
    import keras
    import numpy as np

    def run(*values):
        outs = fn(*[np.asarray(v) for v in values])
        return tuple(np.asarray(o, dtype=dtype).reshape(shape) for o, (shape, dtype) in zip(outs, out_specs))

    if keras.config.backend() == "jax":
        import jax

        specs = tuple(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in out_specs)
        return jax.pure_callback(run, specs, *[jax.lax.stop_gradient(a) for a in args])
    outs = run(*[ops.convert_to_numpy(ops.stop_gradient(a)) for a in args])
    return tuple(ops.convert_to_tensor(o) for o in outs)


def eager_only_placeholder(layer_name: str, *tensors) -> bool:
    """Guards host-side (NumPy) computations with data-dependent output sizes.

    Returns ``True`` during Keras shape inference, where the caller should return a
    placeholder result. Raises inside compiled functions (``tf.function`` / XLA /
    ``jax.jit``), where the computation cannot run. Returns ``False`` when eager.
    """
    try:
        from keras.src.backend.common.symbolic_scope import in_symbolic_scope

        if in_symbolic_scope():
            return True
    except Exception:
        pass
    if any(_is_compiled_trace(t) for t in tensors):
        raise RuntimeError(
            f"{layer_name} computes a data-dependent number of clusters on the host, so it cannot run "
            "inside a compiled function (tf.function, XLA or jax.jit). Compile the model with "
            "`run_eagerly=True`."
        )
    return False


def degree(index, num_nodes: Optional[int] = None, dtype=None):
    """Computes the (in/out) degree of a given index tensor.

    Args:
        index: 1D tensor of node indices.
        num_nodes: The number of nodes.
        dtype: Output data type.
    """
    index = ops.cast(index, "int32")
    if num_nodes is None:
        if is_tracing(index):
            num_nodes = ops.shape(index)[0]
        else:
            num_nodes = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0
    try:
        num_nodes = int(num_nodes)
    except (TypeError, ValueError):
        pass
    ones = ops.ones((ops.shape(index)[0],), dtype=dtype or "float32")
    deg = segment_sum(ones, index, num_segments=num_nodes)
    if dtype is not None:
        deg = ops.cast(deg, dtype)
    return deg


def remove_self_loops(
    edge_index,
    edge_attr=None,
) -> Tuple:
    """Removes self-loops from `edge_index` and optional `edge_attr`."""
    if is_tracing(edge_index):
        return edge_index, edge_attr
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


def remove_self_loops_masked(
    edge_index,
    edge_attr=None,
) -> Tuple:
    """Removes self-loops in a way that is also correct under static-shape tracing.

    Eagerly, self-loops are dropped (as in :func:`remove_self_loops`) and the
    returned mask is ``None``. Under tracing (XLA / ``jax.jit``) the edge count
    must stay static, so all edges are kept and a boolean ``keep_mask`` of shape
    ``[E]`` is returned that is ``False`` at the original self-loops. Callers must
    exclude masked edges from aggregation.

    Returns:
        ``(edge_index, edge_attr, keep_mask)``
    """
    if not is_tracing(edge_index):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return edge_index, edge_attr, None
    edge_index = ops.convert_to_tensor(edge_index)
    return edge_index, edge_attr, ops.not_equal(edge_index[0], edge_index[1])


def extend_mask_for_self_loops(keep_mask, num_nodes):
    """Extends a ``keep_mask`` to cover the ``num_nodes`` loops appended by :func:`add_self_loops`."""
    if keep_mask is None:
        return None
    return ops.concatenate([keep_mask, ops.ones((num_nodes,), dtype="bool")], axis=0)


def mask_edge_logits(alpha, keep_mask):
    """Sets the logits of masked edges to ``-inf`` so they receive zero softmax weight."""
    if keep_mask is None:
        return alpha
    mask = ops.reshape(keep_mask, (-1,) + (1,) * (len(alpha.shape) - 1))
    return ops.where(mask, alpha, float("-inf"))  # a scalar also works on torch's meta device


def add_self_loops(
    edge_index,
    edge_attr=None,
    fill_value: Union[float, str, None] = None,
    num_nodes: Optional[int] = None,
) -> Tuple:
    """Adds self-loops to `edge_index` and optional `edge_attr`."""
    if is_tracing(edge_index) or is_tracing(num_nodes):
        if num_nodes is None:
            return edge_index, edge_attr
    edge_index = ops.convert_to_tensor(edge_index)
    if num_nodes is None:
        num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0
    else:
        try:
            num_nodes = int(num_nodes)
        except (TypeError, ValueError):
            pass

    loop_index = ops.arange(0, num_nodes, dtype=edge_index.dtype)
    loop_index = ops.stack([loop_index, loop_index], axis=0)
    edge_index = ops.concatenate([edge_index, loop_index], axis=1)

    if edge_attr is not None:
        edge_attr = ops.convert_to_tensor(edge_attr)
        attr_shape = (num_nodes,) + tuple(edge_attr.shape[1:]) if hasattr(edge_attr, "shape") else (num_nodes,)
        if fill_value is None:
            loop_attr = ops.zeros(attr_shape, dtype=edge_attr.dtype)
        elif isinstance(fill_value, (int, float)):
            loop_attr = full(attr_shape, fill_value, dtype=edge_attr.dtype)
        elif fill_value == "add" or fill_value == "mean":
            loop_attr = ops.zeros(attr_shape, dtype=edge_attr.dtype)
        else:
            loop_attr = full(attr_shape, fill_value, dtype=edge_attr.dtype)
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
        if is_tracing(edge_index):
            num_nodes = edge_index.shape[1] if hasattr(edge_index, "shape") and edge_index.shape[1] is not None else ops.shape(edge_index)[1]
        else:
            num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0
    try:
        num_nodes = int(num_nodes)
    except (TypeError, ValueError):
        pass

    if edge_weight is None:
        num_edges = edge_index.shape[1] if hasattr(edge_index, "shape") and edge_index.shape[1] is not None else ops.shape(edge_index)[1]
        edge_weight = ops.ones((num_edges,), dtype=dtype or "float32")

    if add_self_loops:
        edge_index, edge_weight = globals()["add_self_loops"](
            edge_index, edge_weight, fill_value=fill_value, num_nodes=num_nodes
        )

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    row_cast = ops.cast(row, "int32")
    col_cast = ops.cast(col, "int32")
    idx_cast = ops.cast(idx, "int32")

    deg = segment_sum(edge_weight, idx_cast, num_segments=num_nodes)
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
        if is_tracing(edge_index):
            num_nodes = edge_index.shape[1] if hasattr(edge_index, "shape") and edge_index.shape[1] is not None else ops.shape(edge_index)[1]
        else:
            num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0
    try:
        num_nodes = int(num_nodes)
    except (TypeError, ValueError):
        pass

    if edge_weight is None:
        num_edges = edge_index.shape[1] if hasattr(edge_index, "shape") and edge_index.shape[1] is not None else ops.shape(edge_index)[1]
        edge_weight = ops.ones((num_edges,), dtype=dtype or "float32")

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


def _infer_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    if hasattr(index, "is_meta") and index.is_meta:
        return None
    try:
        if hasattr(index, "numpy") and not hasattr(index, "_has_symbolic_representation"):
            # NumPy array or eager tensor with numpy()
            import torch
            if not isinstance(index, torch.Tensor):
                return int(index.numpy().max()) + 1 if index.shape[0] > 0 else 0
        if hasattr(index, "max"):
            max_val = index.max()
            if hasattr(max_val, "item"):
                return int(max_val.item()) + 1
            return int(max_val) + 1
        return int(ops.convert_to_numpy(ops.max(index))) + 1 if ops.shape(index)[0] > 0 else 0
    except Exception:
        pass
    try:
        return ops.cast(ops.max(index), "int32") + 1
    except Exception:
        return None


def softmax(src, index, num_nodes: Optional[int] = None, dim: int = -2):
    """Computes a sparsely evaluated softmax over index."""
    index = ops.cast(index, "int32")
    num_nodes = _infer_dim_size(index, num_nodes)

    max_val = segment_max(src, index, num_segments=num_nodes)
    max_val = ops.take(max_val, index, axis=dim)
    exp = ops.exp(src - max_val)
    sum_val = segment_sum(exp, index, num_segments=num_nodes)
    sum_val = ops.take(sum_val, index, axis=dim)
    return exp / (sum_val + 1e-12)


def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """Computes scatter / segment reduction."""
    index = ops.cast(index, "int32")
    dim_size = _infer_dim_size(index, dim_size)

    if reduce in ("add", "sum"):
        return segment_sum(src, index, num_segments=dim_size)
    elif reduce == "mean":
        sum_val = segment_sum(src, index, num_segments=dim_size)
        ones = ops.ones_like(src)
        count = segment_sum(ones, index, num_segments=dim_size)
        count = ops.maximum(count, 1.0)
        return sum_val / count
    elif reduce == "max":
        return segment_max(src, index, num_segments=dim_size)
        val = segment_max(src, index, num_segments=dim_size)
        ones = ops.ones((ops.shape(index)[0], 1), dtype=src.dtype)
        count = segment_sum(ones, index, num_segments=dim_size)
        return ops.where(ops.greater(count, 0), val, ops.zeros_like(val))
    elif reduce == "min":
        return segment_min(src, index, num_segments=dim_size)
        val = segment_min(src, index, num_segments=dim_size)
        ones = ops.ones((ops.shape(index)[0], 1), dtype=src.dtype)
        count = segment_sum(ones, index, num_segments=dim_size)
        return ops.where(ops.greater(count, 0), val, ops.zeros_like(val))
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")



