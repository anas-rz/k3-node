# ported from spektral

import inspect
from keras import layers, ops

from k3_node.utils import (
    is_layer_kwarg,
    deserialize_kwarg,
    deserialize_scatter,
    serialize_scatter,
    serialize_kwarg,
    is_keras_kwarg,
)
from k3_node.ops import get_source_target


class MessagePassing(layers.Layer):
    def __init__(self, aggregate="sum", **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters
        self.agg = deserialize_scatter(aggregate)

    def call(self, inputs, **kwargs):
        x, a, e = self.get_inputs(inputs)
        return self.propagate(x, a, e)

    def build(self, input_shape):
        self.built = True

    def propagate(self, x, a, e=None, **kwargs):
        self.n_nodes = ops.shape(x)[-2]
        self.index_sources, self.index_targets = get_source_target(a)

        msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, **msg_kwargs)

        agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)
        return output

    def message(self, x, **kwargs):
        return self.get_sources(x)

    def aggregate(self, messages, **kwargs):
        return self.agg(messages, self.index_targets, self.n_nodes)

    def update(self, embeddings, **kwargs):
        return embeddings

    def get_targets(self, x):
        return ops.take(x, self.index_targets, axis=-2)

    def get_sources(self, x):
        return ops.take(x, self.index_sources, axis=-2)

    def get_kwargs(self, x, a, e, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "e":
                output[k] = e
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        if len(inputs) == 3:
            x, a, e = inputs
            assert len(ops.shape(e)) in (2, 3), "E must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                "Expected 2 or 3 inputs tensors (X, A, E), got {}.".format(len(inputs))
            )
        assert len(ops.shape(a)) == 2, "A must have rank 2"

        return x, a, e

    @staticmethod
    def preprocess(a):
        return a

    def get_config(self):
        mp_config = {"aggregate": serialize_scatter(self.agg)}
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()

        return {**base_config, **keras_config, **mp_config, **self.config}

    @property
    def config(self):
        return {}
