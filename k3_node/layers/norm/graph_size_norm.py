from keras import layers, ops


class GraphSizeNorm(layers.Layer):
    r"""Applies Graph Size Normalization over each individual graph in a batch
    of node features:

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, batch=None, batch_size=None):
        if batch is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, batch = x
            elif len(x) == 3:
                x, batch, batch_size = x

        if batch is None:
            num_nodes = ops.cast(ops.shape(x)[0], dtype=x.dtype)
            return x * ops.power(num_nodes, -0.5)

        if batch_size is None:
            batch_size = ops.cast(ops.max(batch), "int32") + 1

        batch = ops.cast(batch, "int32")
        ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
        deg = ops.segment_sum(ones, batch, num_segments=batch_size)
        inv_sqrt_deg = ops.power(deg, -0.5)
        scale = ops.take(inv_sqrt_deg, batch, axis=0)
        return x * scale

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape
