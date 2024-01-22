import warnings
from functools import wraps

from keras import ops, backend
from keras.layers import Layer

from k3_node.utils import (
    is_keras_kwarg,
    is_layer_kwarg,
    deserialize_kwarg,
    serialize_kwarg,
)


class Conv(Layer):
    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.supports_masking = True
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)
        self.call = check_dtypes_decorator(self.call)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a


def check_dtypes_decorator(call):
    @wraps(call)
    def _inner_check_dtypes(inputs, **kwargs):
        inputs = check_dtypes(inputs)
        return call(inputs, **kwargs)

    return _inner_check_dtypes


def check_dtypes(inputs):
    for value in inputs:
        if not hasattr(value, "dtype"):
            # It's not a valid tensor.
            return inputs

    if len(inputs) == 2:
        x, a = inputs
        e = None
    elif len(inputs) == 3:
        x, a, e = inputs
    else:
        return inputs

    if backend.is_int_dtype(a.dtype) and backend.is_float_dtype(x.dtype):
        warnings.warn(
            f"The adjacency matrix of dtype {a.dtype} is incompatible with the dtype "
            f"of the node features {x.dtype} and has been automatically cast to "
            f"{x.dtype}."
        )
        a = ops.cast(a, x.dtype)

    output = [_ for _ in [x, a, e] if _ is not None]
    return output
