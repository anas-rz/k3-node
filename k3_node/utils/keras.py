# ported from spektral

from keras import ops, activations, constraints, initializers, regularizers


def segment_softmax(x, indices, n_nodes=None):
    n_nodes = ops.max(indices) + 1 if n_nodes is None else n_nodes
    e_x = ops.exp(x - ops.take(ops.segment_max(x, indices, n_nodes), indices))
    e_x /= ops.take(ops.segment_sum(e_x, indices, n_nodes) + 1e-9, indices)
    return e_x


LAYER_KWARGS = {"activation", "use_bias"}
KERAS_KWARGS = {
    "trainable",
    "name",
    "dtype",
    "dynamic",
    "input_dim",
    "input_shape",
    "batch_input_shape",
    "batch_size",
    "weights",
    "activity_regularizer",
    "autocast",
    "implementation",
}


def is_layer_kwarg(key):
    return key not in KERAS_KWARGS and (
        key.endswith("_initializer")
        or key.endswith("_regularizer")
        or key.endswith("_constraint")
        or key in LAYER_KWARGS
    )


def is_keras_kwarg(key):
    return key in KERAS_KWARGS


def deserialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.get(attr)
    if key.endswith("_regularizer"):
        return regularizers.get(attr)
    if key.endswith("_constraint"):
        return constraints.get(attr)
    if key == "activation":
        return activations.get(attr)
    return attr


def serialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.serialize(attr)
    if key.endswith("_regularizer"):
        return regularizers.serialize(attr)
    if key.endswith("_constraint"):
        return constraints.serialize(attr)
    if key == "activation":
        return activations.serialize(attr)
    if key == "use_bias":
        return attr


OP_DICT = {
    "sum": ops.segment_sum,
    # "mean": scatter_mean,
    "max": ops.segment_max,
    # "min": scatter_min,
    # "prod": scatter_prod,
}


def deserialize_scatter(scatter):
    if isinstance(scatter, str) and scatter in OP_DICT:
        return OP_DICT[scatter]
    elif callable(scatter):
        return scatter
    else:
        raise ValueError(f"scatter must be callable or string: {list(OP_DICT.keys())}")


def serialize_scatter(identifier):
    if identifier in OP_DICT:
        return identifier
    elif hasattr(identifier, "__name__"):
        for k, v in OP_DICT.items():
            if v.__name__ == identifier.__name__:
                return k
        return None
