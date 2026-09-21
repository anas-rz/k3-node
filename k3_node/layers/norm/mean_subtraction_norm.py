from keras import layers, ops


class MeanSubtractionNorm(layers.Layer):
    r"""Applies layer normalization by subtracting the mean from the inputs
    as described in the `"Revisiting 'Over-smoothing' in Deep GCNs"
    <https://arxiv.org/abs/2003.13663>`_ paper.

    .. math::
        \mathbf{x}_i = \mathbf{x}_i - \frac{1}{|\mathcal{V}|}
        \sum_{j \in \mathcal{V}} \mathbf{x}_j
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, batch=None, dim_size=None):
        if batch is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, batch = x
            elif len(x) == 3:
                x, batch, dim_size = x

        if batch is None:
            return x - ops.mean(x, axis=0, keepdims=True)

        if dim_size is not None and not isinstance(dim_size, int):
            try:
                dim_size = int(dim_size)
            except Exception:
                pass
        elif dim_size is None:
            dim_size = ops.cast(ops.max(batch), "int32") + 1

        batch = ops.cast(batch, "int32")
        ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
        counts = ops.maximum(ops.segment_sum(ones, batch, num_segments=dim_size), 1.0)
        mean = ops.segment_sum(x, batch, num_segments=dim_size) / counts
        return x - ops.take(mean, batch, axis=0)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape
