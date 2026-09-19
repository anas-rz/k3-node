from math import ceil
from typing import Optional
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing


class GroupedConv1dFlat(keras.layers.Layer):
    """A 1D convolution over input of shape (N, C, K) with kernel_size=K and groups=C."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.multiplier = out_channels // in_channels

    def build(self, input_shape=None):
        self.kernel = self.add_weight(
            shape=(self.in_channels, self.multiplier, self.kernel_size),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.out_channels,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        super().build(input_shape)

    def call(self, x):
        # x: (N, C, K)
        out = ops.einsum("nck,cmk->ncm", x, self.kernel)
        out = ops.reshape(out, (-1, self.out_channels)) + self.bias
        return out


class XConv(keras.layers.Layer):
    r"""The convolutional operator on :math:`\mathcal{X}`-transformed points
    from the `"PointCNN: Convolution On X-Transformed Points"
    <https://arxiv.org/abs/1801.07791>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Point cloud dimensionality.
        kernel_size (int): Size of the convolving kernel.
        hidden_channels (int, optional): Dimensionality of lifted points.
        dilation (int, optional): Dilation factor. (default: :obj:`1`)
        bias (bool, optional): Whether to learn an additive bias. (default: :obj:`True`)
        num_workers (int, optional): Kept for PyG compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int,
        hidden_channels: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_bias = bias

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        # mlp1
        self.mlp1_l1 = keras.layers.Dense(C_delta)
        self.mlp1_bn1 = keras.layers.BatchNormalization(axis=-1)
        self.mlp1_l2 = keras.layers.Dense(C_delta)
        self.mlp1_bn2 = keras.layers.BatchNormalization(axis=-1)

        # mlp2
        self.mlp2_l1 = keras.layers.Dense(K * K)
        self.mlp2_bn1 = keras.layers.BatchNormalization(axis=-1)
        self.mlp2_conv1 = GroupedConv1dFlat(K, K * K, K)
        self.mlp2_bn2 = keras.layers.BatchNormalization(axis=-1)
        self.mlp2_conv2 = GroupedConv1dFlat(K, K * K, K)
        self.mlp2_bn3 = keras.layers.BatchNormalization(axis=-1)

        # conv
        C_total = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_total))
        self.conv_op = GroupedConv1dFlat(C_total, C_total * depth_multiplier, K)
        self.conv_lin = keras.layers.Dense(C_out, use_bias=bias)

    def _mlp1(self, pos):
        # pos: (N*K, D)
        h = ops.elu(self.mlp1_l1(pos))
        h = self.mlp1_bn1(h)
        h = ops.elu(self.mlp1_l2(h))
        h = self.mlp1_bn2(h)
        return h

    def _mlp2(self, pos_flat):
        # pos_flat: (N, K * D)
        K = self.kernel_size
        h = ops.elu(self.mlp2_l1(pos_flat))
        h = self.mlp2_bn1(h)
        h = ops.reshape(h, (-1, K, K))
        h = ops.elu(self.mlp2_conv1(h))
        h = self.mlp2_bn2(h)
        h = ops.reshape(h, (-1, K, K))
        h = self.mlp2_conv2(h)
        h = self.mlp2_bn3(h)
        return ops.reshape(h, (-1, K, K))

    def _conv(self, x_transformed):
        # x_transformed: (N, C_total, K)
        h = self.conv_op(x_transformed)
        return self.conv_lin(h)

    def call(self, x, pos, batch=None):
        if len(ops.shape(pos)) == 1:
            pos = ops.expand_dims(pos, axis=-1)
        N = ops.shape(pos)[0]
        K = self.kernel_size
        D = self.dim

        # Pairwise distance KNN
        diff = ops.expand_dims(pos, axis=1) - ops.expand_dims(pos, axis=0)
        dist = ops.sum(diff * diff, axis=-1)

        if batch is not None:
            mask = ops.equal(ops.expand_dims(batch, axis=1), ops.expand_dims(batch, axis=0))
            dist = ops.where(mask, dist, 1e10)

        _, top_k = ops.top_k(-dist, k=K * self.dilation, sorted=True)
        if self.dilation > 1:
            top_k = top_k[:, ::self.dilation]

        row = ops.repeat(ops.arange(N), K)
        col = ops.reshape(top_k, (-1,))

        pos_diff = ops.take(pos, col, axis=0) - ops.take(pos, row, axis=0)

        x_star = self._mlp1(pos_diff)
        x_star = ops.reshape(x_star, (N, K, self.hidden_channels))

        if x is not None:
            if len(ops.shape(x)) == 1:
                x = ops.expand_dims(x, axis=-1)
            x_col = ops.take(x, col, axis=0)
            x_col = ops.reshape(x_col, (N, K, self.in_channels))
            x_star = ops.concatenate([x_star, x_col], axis=-1)

        x_star = ops.transpose(x_star, (0, 2, 1))  # (N, C_total, K)

        transform_matrix = self._mlp2(ops.reshape(pos_diff, (N, K * D)))  # (N, K, K)

        x_transformed = ops.matmul(x_star, transform_matrix)  # (N, C_total, K)

        out = self._conv(x_transformed)
        return out

