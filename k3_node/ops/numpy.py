from keras import backend, ops
from k3_node.utils.backend_import import *


def polyval(p, x):
    p = ops.convert_to_tensor(p)

    result = ops.zeros_like(x)

    for i in range(p.shape[0]):
        result = result * x + p[i]

    return result


def get_unique(inputs):
    if backend.backend() == "tensorflow":
        return tf.unique(inputs)
    elif backend.backend() == "torch":
        return torch.unique(inputs, return_inverse=True)
    elif backend.backend() == "jax":
        return jnp.unique(inputs, return_inverse=True)
    elif backend.backend() == "numpy":
        return np.unique(inputs, return_inverse=True)
