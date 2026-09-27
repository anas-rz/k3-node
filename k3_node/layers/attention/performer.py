import keras
from keras import ops, layers


def _orthogonal_matrix(dim: int, seed: int = None):
    # Random matrix from normal distribution
    mat = keras.random.normal((dim, dim), seed=seed)
    # QR decomposition to two orthogonal matrices
    q, _ = ops.qr(mat, mode="reduced")
    return ops.transpose(q, [1, 0])


def orthogonal_matrix(num_rows: int, num_cols: int, seed=None):
    """Function ``orthogonal_matrix``.

    Example:
        ```python
        from k3_node.layers import orthogonal_matrix

        projection = orthogonal_matrix(num_rows=16, num_cols=8)  # random orthogonal features (Performer)
        print(tuple(projection.shape))  # (16, 8)
        ```
    """
    num_full_blocks = int(num_rows / num_cols)
    blocks = []
    for _ in range(num_full_blocks):
        q = _orthogonal_matrix(num_cols)
        blocks.append(q)
    remain_rows = num_rows - num_full_blocks * num_cols
    if remain_rows > 0:
        q = _orthogonal_matrix(num_cols)
        blocks.append(q[:remain_rows])
    mat = ops.concatenate(blocks)
    return mat


def linear_attention(q, k, v):
    # Plain NumPy inputs cannot be mixed with backend tensors (e.g. `ndarray @ torch.Tensor`).
    """Function ``linear_attention``.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import linear_attention

        q = k = v = np.random.rand(1, 2, 10, 8).astype("float32")  # [batch, heads, num_nodes, head_dim]
        print(tuple(linear_attention(q, k, v).shape))  # (1, 2, 10, 8)
        ```
    """
    q, k, v = ops.convert_to_tensor(q), ops.convert_to_tensor(k), ops.convert_to_tensor(v)
    _k = ops.expand_dims(ops.sum(k, axis=-2), axis=-1)
    D_inv = 1.0 / (q @ _k)
    kv = ops.transpose(k, axes=[0, 1, 3, 2]) @ v
    qkv = q @ kv
    out = ops.einsum("...L,...Ld->...Ld", ops.squeeze(D_inv, axis=-1), qkv)
    return out


def generalized_kernel(x, mat, kernel=ops.relu, epsilon=0.001):
    """Function ``generalized_kernel``.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import generalized_kernel, orthogonal_matrix

        x = np.random.rand(1, 2, 10, 8).astype("float32")  # [batch, heads, num_nodes, head_dim]
        features = generalized_kernel(x, orthogonal_matrix(16, 8))  # random-feature map of x
        print(tuple(features.shape))  # (1, 2, 10, 16)
        ```
    """
    batch_size, num_heads = ops.shape(x)[:2]
    projection = ops.transpose(mat, axes=[1, 0])  # Transpose along correct axes
    projection = ops.tile(projection, [1, num_heads, 1, 1])  # Expand dimensions
    x = ops.matmul(x, projection)
    out = kernel(x) + epsilon
    return out


class PerformerProjection(layers.Layer):
    """Layer ``PerformerProjection``.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import PerformerProjection

        q = k = v = np.random.rand(1, 2, 10, 8).astype("float32")  # [batch, heads, num_nodes, head_dim]
        proj = PerformerProjection(num_cols=8)  # random-feature approximation of softmax attention
        print(tuple(proj(q, k, v).shape))  # (1, 2, 10, 8)
        ```
    """
    def __init__(self, num_cols, kernel=ops.relu):
        super().__init__()
        import math
        self.num_rows = max(1, int(num_cols * math.log(max(2, num_cols))))
        self.num_cols = num_cols

        # Generate an orthogonal projection matrix
        self.projection_matrix = orthogonal_matrix(self.num_rows, self.num_cols)
        self.kernel = kernel

    def call(self, q, k, v):
        q = generalized_kernel(q, self.projection_matrix, self.kernel)
        k = generalized_kernel(k, self.projection_matrix, self.kernel)
        out = linear_attention(q, k, v)
        return out


class PerformerAttention(layers.Layer):
    """
    `k3_node.layers.PerformerAttention`
    
    Initialization Arguments:

    Args:
        channels: The number of output channels.
        heads: The number of attention heads.
        head_channels: The number of attention heads.
        kernel: activation function.
        qkv_bias: activation function.
        attn_out_bias: Bias in Attention Out.
        dropout: Dropout rate.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import PerformerAttention

        x = np.random.rand(1, 10, 8).astype("float32")  # [batch, num_nodes, channels]

        mask = np.ones((1, 10), dtype=bool)  # which nodes are real (not padding)
        attn = PerformerAttention(channels=8, heads=2)  # linear-complexity attention
        print(tuple(attn(x, mask).shape))  # (1, 10, 8)
        ```
    """
    def __init__(
        self,
        channels,
        heads,
        head_channels=64,
        kernel=ops.relu,
        qkv_bias=False,
        attn_out_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels
        self.kernel = kernel
        self.fast_attn = PerformerProjection(head_channels, kernel)

        inner_channels = head_channels * heads
        self.q = layers.Dense(inner_channels, use_bias=qkv_bias)
        self.k = layers.Dense(inner_channels, use_bias=qkv_bias)
        self.v = layers.Dense(inner_channels, use_bias=qkv_bias)
        self.attn_out = layers.Dense(channels, use_bias=attn_out_bias)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask=None, training=None):
        B, N, *_ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)

        q = ops.transpose(
            ops.reshape(q, (B, N, self.heads, self.head_channels)), axes=(0, 2, 1, 3)
        )
        k = ops.transpose(
            ops.reshape(k, (B, N, self.heads, self.head_channels)), axes=(0, 2, 1, 3)
        )
        v = ops.transpose(
            ops.reshape(v, (B, N, self.heads, self.head_channels)), axes=(0, 2, 1, 3)
        )

        if mask is not None:
            mask = mask[:, None, :, None]
            v = ops.where(mask, v, ops.zeros_like(v))

        out = self.fast_attn(q, k, v)
        out = ops.transpose(out, axes=(0, 2, 1, 3))  # Transpose back
        out = ops.reshape(out, (B, N, -1))
        out = self.attn_out(out)
        out = self.dropout(out, training=training)
        return out
