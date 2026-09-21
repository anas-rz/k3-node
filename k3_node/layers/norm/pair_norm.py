from keras import layers, ops


class PairNorm(layers.Layer):
    r"""Applies pair normalization over node features as described in the
    `"PairNorm: Tackling Oversmoothing in GNNs"
    <https://arxiv.org/abs/1909.12223>`_ paper.

    .. math::
        \mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
        \sum_{i=1}^n \mathbf{x}_i \\

        \mathbf{x}_i^{\prime} &= s \cdot
        \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
        {\| \mathbf{x}_i^c \|}^2_2}}

    Args:
        scale (float, optional): Scaling factor :math:`s` of normalization.
            (default: :obj:`1.0`)
        scale_individually (bool, optional): If set to :obj:`True`, will
            compute the scaling step as :math:`\mathbf{x}^{\prime}_i = s \cdot
            \frac{\mathbf{x}_i^c}{{\| \mathbf{x}_i^c \|}_2}`.
            (default: :obj:`False`)
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, scale: float = 1.0, scale_individually: bool = False,
                 eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def call(self, x, batch=None, batch_size=None):
        if batch is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, batch = x
            elif len(x) == 3:
                x, batch, batch_size = x

        scale = self.scale

        if batch is None:
            x = x - ops.mean(x, axis=0, keepdims=True)

            if not self.scale_individually:
                mean_sq = ops.mean(ops.sum(ops.power(x, 2), axis=-1))
                return scale * x / ops.sqrt(self.eps + mean_sq)
            else:
                norm = ops.sqrt(ops.sum(ops.power(x, 2), axis=-1, keepdims=True))
                return scale * x / (self.eps + norm)

        if batch_size is not None and not isinstance(batch_size, int):
            try:
                batch_size = int(batch_size)
            except Exception:
                pass
        elif batch_size is None:
            batch_size = ops.cast(ops.max(batch), "int32") + 1

        batch = ops.cast(batch, "int32")
        ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
        counts = ops.maximum(ops.segment_sum(ones, batch, num_segments=batch_size), 1.0)
        mean = ops.segment_sum(x, batch, num_segments=batch_size) / counts
        x = x - ops.take(mean, batch, axis=0)

        if not self.scale_individually:
            sq_sum = ops.sum(ops.power(x, 2), axis=-1, keepdims=True)
            mean_sq = ops.segment_sum(sq_sum, batch, num_segments=batch_size) / counts
            denom = ops.sqrt(self.eps + ops.take(mean_sq, batch, axis=0))
            return scale * x / denom
        else:
            norm = ops.sqrt(ops.sum(ops.power(x, 2), axis=-1, keepdims=True))
            return scale * x / (self.eps + norm)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape

