from keras import layers, ops


class InstanceNorm(layers.Layer):
    r"""Applies instance normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.06450>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
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
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.running_mean = self.add_weight(
                shape=(in_channels,),
                initializer="zeros",
                trainable=False,
                name="running_mean",
            )
            self.running_var = self.add_weight(
                shape=(in_channels,),
                initializer="ones",
                trainable=False,
                name="running_var",
            )
        else:
            self.running_mean = None
            self.running_var = None

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.assign(ops.zeros(self.running_mean.shape, dtype=self.running_mean.dtype))
            self.running_var.assign(ops.ones(self.running_var.shape, dtype=self.running_var.dtype))

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.assign(ops.ones(self.weight.shape, dtype=self.weight.dtype))
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def call(self, x, batch=None, batch_size=None, training=None):
        if batch is None and isinstance(x, (tuple, list)):
            if len(x) == 2:
                x, batch = x
            elif len(x) == 3:
                x, batch, batch_size = x

        is_training = training if training is not None else True

        if batch is None:
            batch = ops.zeros((ops.shape(x)[0],), dtype="int32")
            batch_size = 1
        elif batch_size is None:
            batch_size = ops.cast(ops.max(batch), "int32") + 1

        batch = ops.cast(batch, "int32")

        if is_training or not self.track_running_stats:
            ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
            counts = ops.maximum(ops.segment_sum(ones, batch, num_segments=batch_size), 1.0)
            unbiased_counts = ops.maximum(counts - 1.0, 1.0)

            mean = ops.segment_sum(x, batch, num_segments=batch_size) / counts
            x_c = x - ops.take(mean, batch, axis=0)
            sq_diff = ops.segment_sum(ops.power(x_c, 2), batch, num_segments=batch_size)
            var = sq_diff / counts
            unbiased_var = sq_diff / unbiased_counts

            if is_training and self.track_running_stats:
                m = self.momentum
                cur_mean = ops.mean(mean, axis=0)
                cur_var = ops.mean(unbiased_var, axis=0)
                new_running_mean = (1.0 - m) * self.running_mean + m * cur_mean
                new_running_var = (1.0 - m) * self.running_var + m * cur_var
                self.running_mean.assign(new_running_mean)
                self.running_var.assign(new_running_var)

            std_x = ops.take(ops.sqrt(var + self.eps), batch, axis=0)
            out = x_c / std_x
        else:
            x_c = x - self.running_mean
            std = ops.sqrt(self.running_var + self.eps)
            out = x_c / std

        if self.affine:
            out = out * self.weight + self.bias

        return out

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)) and isinstance(input_shape[0], (tuple, list)):
            return input_shape[0]
        return input_shape
