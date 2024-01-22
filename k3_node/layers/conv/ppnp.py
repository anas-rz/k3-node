# ported from stellargraph
from keras.layers import Layer
from keras import ops


class PPNPPropagationLayer(Layer):
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
        features, *As = inputs
        batch_dim, n_nodes, _ = ops.shape(features)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Remove singleton batch dimension
        features = ops.squeeze(features, 0)

        # Propagate the features
        A = As[0]
        output = ops.dot(A, features)

        # Add batch dimension back if we removed it
        if batch_dim == 1:
            output = ops.expand_dims(output, 0)

        return output
