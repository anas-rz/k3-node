from typing import Optional
from keras import ops


from k3_node.layers.conv.utils import is_tracing


def _infer_size(batch, size=None):
    if size is not None:
        return size
    if is_tracing(batch) or (hasattr(batch, "is_meta") and batch.is_meta):
        import keras
        if keras.config.backend() == "jax":
            return 1
        return None
    try:
        if hasattr(batch, "numpy") and not hasattr(batch, "_has_symbolic_representation"):
            import torch
            if not isinstance(batch, torch.Tensor):
                return int(batch[-1]) + 1
        if hasattr(batch, "__getitem__"):
            val = batch[-1]
            if hasattr(val, "item"):
                return int(val.item()) + 1
            return int(val) + 1
        return int(ops.convert_to_numpy(batch[-1])) + 1
    except Exception:
        pass
    try:
        return ops.cast(batch[-1], "int32") + 1
    except Exception:
        return None


def global_add_pool(x, batch: Optional[any] = None, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension.
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
    return ops.segment_sum(x, batch, num_segments=size)


def global_mean_pool(x, batch: Optional[any] = None, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension.
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
    sum_val = ops.segment_sum(x, batch, num_segments=size)
    ones = ops.ones_like(x)
    count = ops.segment_sum(ones, batch, num_segments=size)
    return sum_val / ops.maximum(count, 1.0)


def global_max_pool(x, batch: Optional[any] = None, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension.
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
    return ops.segment_max(x, batch, num_segments=size)
