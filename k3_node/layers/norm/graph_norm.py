from keras import layers, ops


class GraphNorm(layers.Layer):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.eps = eps

        self.weight = self.add_weight(
            shape=(in_channels,),
            initializer="ones",
            trainable=True,
            name="weight",
        )
        self.bias = self.add_weight(
            shape=(in_channels,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        self.mean_scale = self.add_weight(
            shape=(in_channels,),
            initializer="ones",
            trainable=True,
            name="mean_scale",
        )

    def reset_parameters(self):
        self.weight.assign(ops.ones(self.weight.shape, dtype=self.weight.dtype))
        self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))
        self.mean_scale.assign(ops.ones(self.mean_scale.shape, dtype=self.mean_scale.dtype))

    def call(self, x, batch=None, batch_size=None):
        if batch is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, batch = x
            elif len(x) == 3:
                x, batch, batch_size = x

        if batch is None:
            mean = ops.mean(x, axis=0, keepdims=True)
            out = x - mean * self.mean_scale
            var = ops.mean(ops.power(out, 2), axis=0, keepdims=True)
            std = ops.sqrt(var + self.eps)
            return self.weight * out / std + self.bias

        if batch_size is None:
            batch_size = ops.cast(ops.max(batch), "int32") + 1

        batch = ops.cast(batch, "int32")
        ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
        counts = ops.maximum(ops.segment_sum(ones, batch, num_segments=batch_size), 1.0)
        mean = ops.segment_sum(x, batch, num_segments=batch_size) / counts
        out = x - ops.take(mean, batch, axis=0) * self.mean_scale
        var = ops.segment_sum(ops.power(out, 2), batch, num_segments=batch_size) / counts
        std = ops.take(ops.sqrt(var + self.eps), batch, axis=0)
        return self.weight * out / std + self.bias

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape
