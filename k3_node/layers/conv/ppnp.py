# ported from stellargraph
from keras.layers import Layer
from keras import ops


class PPNPPropagation(Layer):
    """
    `k3_node.layers.PPNPPropagation`
    Implementation of PPNP layer

    Args:
        units: Positive integer, dimensionality of the output space.
        final_layer: Deprecated, use tf.gather or GatherIndices instead.
        input_dim: Deprecated, use `keras.layers.Input` with `input_shape` instead.
        **kwargs: Additional arguments to pass to the `Layer` superclass. 
    """
    def __init__(self, units, final_layer=None, input_dim=None, **kwargs):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        super().__init__(**kwargs)

        self.units = units
        if final_layer is not None:
            raise ValueError("'final_layer' is not longer supported.")

    def get_config(self):
        config = {"units": self.units}

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        feature_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        self.built = True

    def call(self, inputs):
        x, a = inputs
        n_nodes, _ = ops.shape(x)
        output = ops.dot(x, a)
        return output
