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