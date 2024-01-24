from keras import backend


if backend.backend() == "tensorflow":
    import tensorflow as tf
elif backend.backend() == "torch":
    import torch
elif backend.backend() == "jax":
    import jax.numpy as jnp
elif backend.backend() == "numpy":
    import numpy as np
else:
    raise NotImplementedError


def polyval(p, x):
    """
    Evaluate a polynomial at a point x.
    """
    if backend.backend() == "tensorflow":
        return tf.math.polyval(p, x)
    elif backend.backend() == "torch":
        raise NotImplementedError
        return torch.polyval(p, x)
    elif backend.backend() == "jax":
        return jnp.polyval(p, x)
    elif backend.backend() == "numpy":
        return np.polyval(p, x)
    else:
        raise NotImplementedError
