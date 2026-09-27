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
