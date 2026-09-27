from typing import Optional
from keras import ops


from k3_node.layers.conv.utils import is_tracing
from k3_node.ops.segment import segment_max, segment_sum
from k3_node.ops.host import to_numpy


def _infer_size(batch, size=None):
    if size is not None:
        return size
    import keras
    from keras.src.backend.common.symbolic_scope import in_symbolic_scope

    if in_symbolic_scope() or (hasattr(batch, "is_meta") and batch.is_meta):
        # Keras shape inference on placeholder values: any size gives the right output rank.
        return 1 if keras.config.backend() == "jax" else None
    if is_tracing(batch):
        if keras.config.backend() == "jax":
            # The number of graphs depends on the values in `batch`, which are unknown inside
            # jax.jit; guessing would silently merge every graph into one.
            raise ValueError(
                "Cannot infer the number of graphs from `batch` inside a compiled JAX function. "
                "Pass `size=` to the pooling function (or `batch_size=` to the model)."
            )
        # TensorFlow graph mode supports a dynamic number of segments.
        return ops.cast(ops.max(batch), "int32") + 1
    # Use the maximum (as PyG does), not the last entry: pooling layers such as EdgePooling
    # return batch vectors that are not sorted by graph.
    try:
        if ops.shape(batch)[0] == 0:
            return 0
        return int(to_numpy(batch).max()) + 1
    except Exception:
        pass
    try:
        return ops.cast(ops.max(batch), "int32") + 1
    except Exception:
        return None


def global_add_pool(x, batch: Optional[any] = None, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import global_add_pool

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        batch = np.repeat([0, 1], 5)  # nodes 0-4 belong to graph 0, nodes 5-9 to graph 1

        out = global_add_pool(x, batch)  # one row per graph
        print(tuple(out.shape))  # (2, 8)
        ```
    """
    if batch is None:
        dim = -1 if len(ops.shape(x)) == 1 else 0
        keepdims = len(ops.shape(x)) <= 2
        out = ops.sum(x, axis=dim, keepdims=keepdims)
        if len(ops.shape(x)) == 1:
            out = ops.reshape(out, (1, 1))
        return out

    batch = ops.cast(batch, dtype="int32")
    if batch.shape[0] is not None and batch.shape[0] == 0:
        num_seg = size if size is not None else 0
        return ops.zeros((num_seg,) + tuple(ops.shape(x)[1:]), dtype=x.dtype)
    size = _infer_size(batch, size)
    return segment_sum(x, batch, num_segments=size)


def global_mean_pool(x, batch: Optional[any] = None, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import global_mean_pool

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        batch = np.repeat([0, 1], 5)  # nodes 0-4 belong to graph 0, nodes 5-9 to graph 1

        out = global_mean_pool(x, batch)  # one row per graph
        print(tuple(out.shape))  # (2, 8)
        ```
    """
    if batch is None:
        dim = -1 if len(ops.shape(x)) == 1 else 0
        keepdims = len(ops.shape(x)) <= 2
        out = ops.mean(x, axis=dim, keepdims=keepdims)
        if len(ops.shape(x)) == 1:
            out = ops.reshape(out, (1, 1))
        return out

    batch = ops.cast(batch, dtype="int32")
    if batch.shape[0] is not None and batch.shape[0] == 0:
        num_seg = size if size is not None else 0
        return ops.zeros((num_seg,) + tuple(ops.shape(x)[1:]), dtype=x.dtype)
    size = _infer_size(batch, size)
    sum_val = segment_sum(x, batch, num_segments=size)
    ones = ops.ones_like(x)
    count = segment_sum(ones, batch, num_segments=size)
    return sum_val / ops.maximum(count, 1.0)


def global_max_pool(x, batch: Optional[any] = None, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import global_max_pool

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        batch = np.repeat([0, 1], 5)  # nodes 0-4 belong to graph 0, nodes 5-9 to graph 1

        out = global_max_pool(x, batch)  # one row per graph
        print(tuple(out.shape))  # (2, 8)
        ```
    """
    if batch is None:
        dim = -1 if len(ops.shape(x)) == 1 else 0
        keepdims = len(ops.shape(x)) <= 2
        out = ops.max(x, axis=dim, keepdims=keepdims)
        if len(ops.shape(x)) == 1:
            out = ops.reshape(out, (1, 1))
        return out

    batch = ops.cast(batch, dtype="int32")
    if batch.shape[0] is not None and batch.shape[0] == 0:
        num_seg = size if size is not None else 0
        return ops.zeros((num_seg,) + tuple(ops.shape(x)[1:]), dtype=x.dtype)
    size = _infer_size(batch, size)
    return segment_max(x, batch, num_segments=size)
