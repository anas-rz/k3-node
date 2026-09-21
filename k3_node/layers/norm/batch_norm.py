from typing import Optional
from keras import layers, ops


class BatchNorm(layers.Layer):
    r"""Applies batch normalization over a batch of features as described in
    the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
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
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
        allow_single_element (bool, optional): If set to :obj:`True`, batches
            with only a single element will work as during in evaluation.
            That is the running mean and variance will be used.
            Requires :obj:`track_running_stats=True`. (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        allow_single_element: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if allow_single_element and not track_running_stats:
            raise ValueError("'allow_single_element' requires "
                             "'track_running_stats' to be set to `True`")

        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.allow_single_element = allow_single_element

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
            self.num_batches_tracked = self.add_weight(
                shape=(),
                initializer="zeros",
                dtype="int32",
                trainable=False,
                name="num_batches_tracked",
            )
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.assign(ops.zeros(self.running_mean.shape, dtype=self.running_mean.dtype))
            self.running_var.assign(ops.ones(self.running_var.shape, dtype=self.running_var.dtype))
            self.num_batches_tracked.assign(ops.cast(0, "int32"))

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.assign(ops.ones(self.weight.shape, dtype=self.weight.dtype))
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def call(self, x, training=None):
        num_samples_static = x.shape[0]
        is_training = training if training is not None else True

        if is_training:
            if num_samples_static is not None and num_samples_static <= 1:
                if not self.allow_single_element:
                    raise ValueError(f"Expected more than 1 value per channel when training, got input size {ops.shape(x)}")
                # Evaluation behavior with running stats
                mean = self.running_mean
                var = self.running_var
            else:
                mean = ops.mean(x, axis=0)
                var = ops.var(x, axis=0)

                if self.track_running_stats:
                    n = ops.cast(ops.shape(x)[0], dtype=x.dtype)
                    unbiased_var = var * n / ops.maximum(n - 1.0, 1.0)
                    if self.momentum is None:
                        count = ops.cast(self.num_batches_tracked + 1, dtype=x.dtype)
                        m = 1.0 / count
                    else:
                        m = self.momentum

                    new_running_mean = (1.0 - m) * self.running_mean + m * mean
                    new_running_var = (1.0 - m) * self.running_var + m * unbiased_var
                    self.running_mean.assign(new_running_mean)
                    self.running_var.assign(new_running_var)
                    self.num_batches_tracked.assign(self.num_batches_tracked + 1)
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = ops.mean(x, axis=0)
                var = ops.var(x, axis=0)

        out = (x - mean) / ops.sqrt(var + self.eps)

        if self.affine:
            out = out * self.weight + self.bias

        return out

    def compute_output_shape(self, input_shape):
        return input_shape


class HeteroBatchNorm(layers.Layer):
    r"""Applies batch normalization over a batch of heterogeneous features as
    described in the `"Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper.
    Compared to :class:`BatchNorm`, :class:`HeteroBatchNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = self.add_weight(
                shape=(num_types, in_channels),
                initializer="ones",
                trainable=True,
                name="weight",
            )
            self.bias = self.add_weight(
                shape=(num_types, in_channels),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.running_mean = self.add_weight(
                shape=(num_types, in_channels),
                initializer="zeros",
                trainable=False,
                name="running_mean",
            )
            self.running_var = self.add_weight(
                shape=(num_types, in_channels),
                initializer="ones",
                trainable=False,
                name="running_var",
            )
            self.num_batches_tracked = self.add_weight(
                shape=(),
                initializer="zeros",
                dtype="int32",
                trainable=False,
                name="num_batches_tracked",
            )
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.assign(ops.zeros(self.running_mean.shape, dtype=self.running_mean.dtype))
            self.running_var.assign(ops.ones(self.running_var.shape, dtype=self.running_var.dtype))
            self.num_batches_tracked.assign(ops.cast(0, "int32"))

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.assign(ops.ones(self.weight.shape, dtype=self.weight.dtype))
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def call(self, x, type_vec=None, training=None):
        if type_vec is None and isinstance(x, (tuple, list)):
            x, type_vec = x

        is_training = training if training is not None else True
        type_vec = ops.cast(type_vec, "int32")

        if not is_training and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            ones = ops.ones((ops.shape(x)[0], 1), dtype=x.dtype)
            counts = ops.maximum(ops.segment_sum(ones, type_vec, num_segments=self.num_types), 1.0)
            mean = ops.segment_sum(x, type_vec, num_segments=self.num_types) / counts
            x_c = x - ops.take(mean, type_vec, axis=0)
            var = ops.segment_sum(ops.power(x_c, 2), type_vec, num_segments=self.num_types) / counts

        if is_training and self.track_running_stats:
            if self.momentum is None:
                count = ops.cast(self.num_batches_tracked + 1, dtype=x.dtype)
                exp_avg_factor = 1.0 / count
            else:
                exp_avg_factor = self.momentum

            new_running_mean = (1.0 - exp_avg_factor) * self.running_mean + exp_avg_factor * mean
            new_running_var = (1.0 - exp_avg_factor) * self.running_var + exp_avg_factor * var
            self.running_mean.assign(new_running_mean)
            self.running_var.assign(new_running_var)
            self.num_batches_tracked.assign(self.num_batches_tracked + 1)

        std = ops.sqrt(var + self.eps)
        mean_taken = ops.take(mean, type_vec, axis=0)
        std_taken = ops.take(std, type_vec, axis=0)
        out = (x - mean_taken) / std_taken

        if self.affine:
            w_taken = ops.take(self.weight, type_vec, axis=0)
            b_taken = ops.take(self.bias, type_vec, axis=0)
            out = out * w_taken + b_taken

        return out

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            return input_shape[0]
        return input_shape
