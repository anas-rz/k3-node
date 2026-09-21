# ported from stellargraph
from keras import layers, ops
from keras.layers import Layer


class PPNPPropagation(Layer):
    """
    `k3_node.layers.PPNPPropagation`
    Implementation of PPNP layer

    Args:
        units: Positive integer, dimensionality of the output space.
        final_layer: Deprecated.
        input_dim: Deprecated.
        **kwargs: Additional arguments to pass to the `Layer` superclass. 
    """
    def __init__(self, units=None, final_layer=None, input_dim=None, **kwargs):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        super().__init__(**kwargs)

        self.units = units
        if units is not None:
            self.dense = layers.Dense(units)
        else:
            self.dense = None

        if final_layer is not None:
            raise ValueError("'final_layer' is no longer supported.")

    def get_config(self):
        config = {"units": self.units}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        if isinstance(input_shapes, (list, tuple)):
            feature_shape = input_shapes[0]
        else:
            feature_shape = input_shapes
        out_dim = self.units if self.units is not None else feature_shape[-1]
        return (*feature_shape[:-1], out_dim)

    def build(self, input_shape=None):
        if self.dense is not None and not getattr(self.dense, "built", False):
                if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                    in_s = input_shape[0]
                else:
                    in_s = input_shape
                self.dense.build(in_s)
        self.built = True

    def call(self, inputs, a=None):
        if a is not None:
            x = inputs
        elif isinstance(inputs, (list, tuple)):
            x, a = inputs[0], inputs[1]
        else:
            x = inputs

        if self.dense is not None:
            x = self.dense(x)

        if a is not None:
            a_shape = getattr(a, "shape", None)
            if a_shape is not None and len(a_shape) == 2 and a_shape[0] == 2:
                N = ops.shape(x)[0]
                indices = ops.transpose(ops.cast(a, "int32"))
                dense_a = ops.scatter_update(ops.zeros((N, N), dtype=x.dtype), indices, ops.ones((ops.shape(indices)[0],), dtype=x.dtype))
                deg = ops.sum(dense_a, axis=-1, keepdims=True)
                deg_inv = ops.where(deg > 0, 1.0 / ops.maximum(deg, 1.0), ops.zeros_like(deg))
                dense_a = dense_a * deg_inv
                return ops.matmul(dense_a, x)
            return ops.matmul(a, x)
        return x
