import numpy as np
from scipy.spatial.distance import cdist
from keras import initializers, layers, ops

from .batch_norm import BatchNorm


class DiffGroupNorm(layers.Layer):
    r"""The differentiable group normalization layer from the `"Towards Deeper
    Graph Neural Networks with Differentiable Group Normalization"
    <https://arxiv.org/abs/2006.06972>`_ paper, which normalizes node features
    group-wise via a learnable soft cluster assignment.

    .. math::

        \mathbf{S} = \text{softmax} (\mathbf{X} \mathbf{W})

    where :math:`\mathbf{W} \in \mathbb{R}^{F \times G}` denotes a trainable
    weight matrix mapping each node into one of :math:`G` clusters.
    Normalization is then performed group-wise via:

    .. math::

        \mathbf{X}^{\prime} = \mathbf{X} + \lambda \sum_{i = 1}^G
        \text{BatchNorm}(\mathbf{S}[:, i] \odot \mathbf{X})

    Args:
        in_channels (int): Size of each input sample :math:`F`.
        groups (int): The number of groups :math:`G`.
        lamda (float, optional): The balancing factor :math:`\lambda` between
            input embeddings and normalized embeddings. (default: :obj:`0.01`)
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
        groups: int,
        lamda: float = 0.01,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.groups = groups
        self.lamda = lamda
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.lin_weight = self.add_weight(
            shape=(in_channels, groups),
            initializer="glorot_uniform",
            trainable=True,
            name="lin_weight",
        )
        self.norm = BatchNorm(
            groups * in_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            name="norm",
        )

    def reset_parameters(self):
        self.lin_weight.assign(
            initializers.GlorotUniform()(shape=self.lin_weight.shape, dtype=self.lin_weight.dtype)
        )
        self.norm.reset_parameters()

    def call(self, x, training=None):
        F, G = self.in_channels, self.groups

        s = ops.softmax(ops.matmul(x, self.lin_weight), axis=-1)  # [N, G]
        out = ops.expand_dims(s, axis=-1) * ops.expand_dims(x, axis=-2)  # [N, G, F]
        out_flat = ops.reshape(out, (-1, G * F))
        out_norm = self.norm(out_flat, training=training)
        out = ops.sum(ops.reshape(out_norm, (-1, G, F)), axis=-2)  # [N, F]

        return x + self.lamda * out

    @staticmethod
    def group_distance_ratio(x, y, eps: float = 1e-5) -> float:
        r"""Measures the ratio of inter-group distance over intra-group
        distance.
        """
        x_np = ops.convert_to_numpy(x)
        y_np = ops.convert_to_numpy(y).astype(np.int64)

        num_classes = int(y_np.max()) + 1

        numerator = 0.0
        for i in range(num_classes):
            mask = (y_np == i)
            if not np.any(mask) or np.all(mask):
                continue
            dist = cdist(x_np[mask], x_np[~mask])
            numerator += (1.0 / dist.size) * float(dist.sum())
        numerator *= 1.0 / ((num_classes - 1) ** 2)

        denominator = 0.0
        for i in range(num_classes):
            mask = (y_np == i)
            if not np.any(mask):
                continue
            dist = cdist(x_np[mask], x_np[mask])
            denominator += (1.0 / dist.size) * float(dist.sum())
        denominator *= 1.0 / num_classes

        return float(numerator / (denominator + eps))

    def compute_output_shape(self, input_shape):
        return input_shape

