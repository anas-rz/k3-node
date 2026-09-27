"""Filled-tensor creation that also works on torch's ``meta`` device.

``keras.ops.full`` / ``keras.ops.full_like`` call ``Tensor.item()`` on the torch backend, which
fails on the ``meta`` device Keras uses to infer shapes before the first training step (the
fallback then prints a long "unbuilt state" warning). ``ones * value`` avoids that.
"""
from keras import ops


def full(shape, fill_value, dtype=None):
    r"""Like :func:`keras.ops.full`."""
    return ops.ones(shape, dtype=dtype) * ops.cast(fill_value, dtype or "float32")


def full_like(x, fill_value, dtype=None):
    r"""Like :func:`keras.ops.full_like`."""
    return ops.ones_like(x, dtype=dtype) * ops.cast(fill_value, dtype or x.dtype)


def scatter(indices, values, shape):
    r"""Like :func:`keras.ops.scatter` (values at duplicate indices are summed)."""
    from keras import backend

    if backend.backend() == "torch":
        import torch

        values = values if torch.is_tensor(values) else ops.convert_to_tensor(values)
        indices = indices if torch.is_tensor(indices) else ops.convert_to_tensor(indices)
        out = torch.zeros(tuple(int(s) for s in shape), dtype=values.dtype, device=values.device)
        return out.index_put_(tuple(indices.long().T), values, accumulate=True)
    return ops.scatter(indices, values, shape)


def repeat(x, repeats, axis=None):
    r"""Like :func:`keras.ops.repeat`; an integer ``repeats`` also works on torch's ``meta`` device."""
    from keras import backend

    if backend.backend() == "torch" and isinstance(repeats, int):
        import torch

        x = x if torch.is_tensor(x) else ops.convert_to_tensor(x)
        return torch.repeat_interleave(x.reshape(-1) if axis is None else x, repeats, dim=axis)
    return ops.repeat(x, repeats, axis=axis)
