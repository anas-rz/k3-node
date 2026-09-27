"""Segment reductions (``segment_sum`` / ``segment_max`` / ``segment_min`` / ``segment_prod``).

These match ``keras.ops.segment_*``. On the torch backend they are reimplemented so they also
work on the ``meta`` device, which Keras uses to infer output shapes before the first training
step: ``keras.ops.segment_*`` fails there (it repeats indices by a tensor-valued count), and the
fallback it triggers runs on uninitialized memory, producing spurious "index out of range"
warnings and occasionally huge allocations.
"""
from keras import backend, ops


def _torch_segment(data, segment_ids, reduction, num_segments):
    import torch

    def as_tensor(x):
        # keras.ops.convert_to_tensor would move tensors to the default device (a meta tensor
        # becomes uninitialized memory), so keep existing tensors where they are.
        if hasattr(x, "value") and not torch.is_tensor(x):
            x = x.value
        return x if torch.is_tensor(x) else ops.convert_to_tensor(x)

    data, segment_ids = as_tensor(data), as_tensor(segment_ids).long()
    if num_segments is None:
        num_segments = int(segment_ids.max()) + 1 if segment_ids.numel() > 0 else 0
    num_segments = int(num_segments)

    # Out-of-range ids go to an extra segment that is dropped at the end (as in Keras).
    segment_ids = torch.where((segment_ids >= 0) & (segment_ids < num_segments), segment_ids, num_segments)
    index = segment_ids.reshape((-1,) + (1,) * (data.dim() - 1)).expand(data.shape)
    fill = {"sum": 0.0, "amax": float("-inf"), "amin": float("inf"), "prod": 1.0}[reduction]
    result = torch.full((num_segments + 1,) + tuple(data.shape[1:]), fill, device=data.device)
    result = result.scatter_reduce(0, index, data.float(), reduction)
    return result[:-1].to(data.dtype)


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    r"""Sums ``data`` over segments; like :func:`keras.ops.segment_sum`."""
    if backend.backend() == "torch":
        return _torch_segment(data, segment_ids, "sum", num_segments)
    return ops.segment_sum(data, segment_ids, num_segments=num_segments, sorted=sorted)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    r"""Maximum of ``data`` over segments; like :func:`keras.ops.segment_max`."""
    if backend.backend() == "torch":
        return _torch_segment(data, segment_ids, "amax", num_segments)
    return ops.segment_max(data, segment_ids, num_segments=num_segments, sorted=sorted)


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    r"""Minimum of ``data`` over segments; like ``keras.ops.segment_min``."""
    if backend.backend() == "torch":
        return _torch_segment(data, segment_ids, "amin", num_segments)
    return ops.segment_min(data, segment_ids, num_segments=num_segments, sorted=sorted)
