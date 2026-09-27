import numpy as np
from keras import ops


def _is_meta(x) -> bool:
    return getattr(getattr(x, "device", None), "type", None) == "meta"


def to_numpy(x):
    """``keras.ops.convert_to_numpy`` for code that runs on the host (NumPy).

    During Keras' shape inference on the torch backend, tensors live on the "meta" device and have
    no values. Host code then gets zeros of the right shape and dtype: only the output shapes matter
    there, not the values.
    """
    if x is None or isinstance(x, np.ndarray):
        return x
    if _is_meta(x):
        import torch

        return np.zeros(tuple(x.shape), dtype=torch.empty(0, dtype=x.dtype).numpy().dtype)
    try:
        return ops.convert_to_numpy(x)
    except Exception:
        # TensorFlow and JAX trace symbolic tensors during Keras' shape inference
        shape = tuple(x.shape)
        if _in_shape_inference() and all(isinstance(d, int) for d in shape):
            import keras

            return np.zeros(shape, dtype=keras.backend.standardize_dtype(x.dtype))
        raise


def _in_shape_inference() -> bool:
    try:
        from keras.src.backend.common.symbolic_scope import in_symbolic_scope

        return in_symbolic_scope()
    except ImportError:
        return False
