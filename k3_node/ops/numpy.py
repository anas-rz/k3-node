from keras import backend
from k3_node.utils.backend_import import *

def polyval(p, x):
    if backend.backend() == "tensorflow":
        return tf.math.polyval(p, x)
    elif backend.backend() == "torch":
        raise NotImplementedError

    elif backend.backend() == "jax":
        return jnp.polyval(p, x)
    elif backend.backend() == "numpy":
        return np.polyval(p, x)
    else:
        raise NotImplementedError
