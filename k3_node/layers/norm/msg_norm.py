from keras import layers, ops


class MessageNorm(layers.Layer):
    r"""Applies message normalization over the aggregated messages as described
    in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::

        \mathbf{x}_i^{\prime} = \mathbf{x}_{i} + s \cdot
        {\| \mathbf{x}_i \|}_2 \cdot
        \frac{\mathbf{m}_{i}}{{\|\mathbf{m}_i\|}_2}

    Args:
        learn_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor :math:`s` of message normalization.
            (default: :obj:`False`)
    """
    def __init__(self, learn_scale: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.learn_scale = learn_scale
        self.scale = self.add_weight(
            shape=(1,),
            initializer="ones",
            trainable=learn_scale,
            name="scale",
        )

    def reset_parameters(self):
        self.scale.assign(ops.ones(self.scale.shape, dtype=self.scale.dtype))

    def call(self, x, msg=None, p=2.0):
        if msg is None and isinstance(x, (tuple, list)):
            x, msg = x

        msg_norm = ops.maximum(ops.norm(msg, ord=p, axis=-1, keepdims=True), 1e-12)
        msg_normalized = msg / msg_norm
        x_norm = ops.norm(x, ord=p, axis=-1, keepdims=True)
        return msg_normalized * x_norm * self.scale

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            return input_shape[0]
        return input_shape
