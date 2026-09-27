from typing import Optional, Tuple
from k3_node.layers.aggr.base import to_dense_batch
from keras import initializers, layers, ops

EPS = 1e-15


class MemPooling(layers.Layer):
    r"""Memory based pooling layer from `"Memory-Based Graph Networks"
    <https://arxiv.org/abs/2002.09518>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import MemPooling

        x = np.random.rand(2, 5, 8).astype("float32")  # dense batch: 2 graphs with 5 nodes each
        layer = MemPooling(in_channels=8, out_channels=16, heads=2, num_clusters=3)
        x_pool, assignment = layer(x)
        print(tuple(x_pool.shape))  # (2, 3, 16)
        print(tuple(assignment.shape))  # (2, 5, 3): soft assignment of nodes to clusters
        ```
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        num_clusters: int,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_clusters = num_clusters
        self.tau = tau

        self.k = self.add_weight(
            shape=(heads, num_clusters, in_channels),
            initializer=initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
            name="k",
        )
        self.conv_weight = self.add_weight(
            shape=(heads, 1),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            name="conv_weight",
        )
        self.lin = layers.Dense(out_channels, use_bias=False, name="lin")

    def build(self, input_shape=None):
        self.lin.build((None, self.num_clusters, self.in_channels))
        super().build(input_shape)

    def reset_parameters(self):
        init = initializers.RandomUniform(minval=-1.0, maxval=1.0)
        self.k.assign(init(self.k.shape, dtype=self.k.dtype))
        glorot = initializers.GlorotUniform()
        self.conv_weight.assign(glorot(self.conv_weight.shape, dtype=self.conv_weight.dtype))

    @staticmethod
    def kl_loss(S) -> any:
        r"""The additional KL divergence-based loss."""
        S_2 = ops.power(S, 2)
        P = S_2 / ops.sum(S, axis=1, keepdims=True)
        denom = ops.sum(P, axis=2, keepdims=True)
        denom = ops.where(ops.sum(S, axis=2, keepdims=True) == 0.0, 1.0, denom)
        P = P / denom

        S_clamped = ops.maximum(S, EPS)
        P_clamped = ops.maximum(P, EPS)
        # KL(P || S) = sum(P * (log(P) - log(S)))
        kl = P * (ops.log(P_clamped) - ops.log(S_clamped))
        return ops.mean(ops.sum(kl, axis=(1, 2)))

    def call(
        self,
        x,
        batch: Optional[any] = None,
        mask: Optional[any] = None,
        max_num_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[any, any]:
        r"""Forward pass."""
        if len(ops.shape(x)) == 2:
            # Node-level input: one dense [num_nodes, channels] block per graph (as in PyG)
            if batch is None:
                batch = ops.zeros((ops.shape(x)[0],), dtype="int32")
            x, mask = to_dense_batch(x, ops.cast(batch, "int32"), dim_size=batch_size,
                                     max_num_elements=max_num_nodes)
        elif mask is None:
            mask = ops.ones((ops.shape(x)[0], ops.shape(x)[1]), dtype=bool)

        B = ops.shape(x)[0]
        N = ops.shape(x)[1]
        H = self.heads
        K = self.num_clusters

        # Compute pairwise squared Euclidean distance between k and x
        # k: [H, K, C], x: [B, N, C]
        # Reshape to [H * K, C] and [B * N, C]
        k_flat = ops.reshape(self.k, (H * K, self.in_channels))
        x_flat = ops.reshape(x, (B * N, self.in_channels))

        k_sq = ops.sum(ops.power(k_flat, 2), axis=-1, keepdims=True)  # [HK, 1]
        x_sq = ops.sum(ops.power(x_flat, 2), axis=-1, keepdims=True)  # [BN, 1]
        dot = ops.matmul(k_flat, ops.transpose(x_flat))  # [HK, BN]
        dist = ops.maximum(k_sq + ops.transpose(x_sq) - 2.0 * dot, 0.0)  # [HK, BN]

        dist = ops.power(1.0 + dist / self.tau, -(self.tau + 1.0) / 2.0)
        # Reshape to [H, K, B, N] then permute to [B, H, N, K]
        dist = ops.reshape(dist, (H, K, B, N))
        dist = ops.transpose(dist, (2, 0, 3, 1))

        S = dist / ops.sum(dist, axis=-1, keepdims=True)  # [B, H, N, K]

        # Conv over head dimension: reduce H -> 1
        # conv_weight: [H, 1]
        # Transpose S to [B, N, K, H] and multiply by [H, 1]
        S_perm = ops.transpose(S, (0, 2, 3, 1))  # [B, N, K, H]
        S_conv = ops.squeeze(ops.matmul(S_perm, self.conv_weight), axis=-1)  # [B, N, K]
        # [B, N, K]; with a single cluster every node belongs to it fully
        S = ops.softmax(S_conv, axis=-1) if K > 1 else ops.ones_like(S_conv)

        mask_f = ops.cast(ops.reshape(mask, (B, N, 1)), S.dtype)
        S = S * mask_f

        # x_out: [B, K, out_channels]
        # S.transpose(1, 2) is [B, K, N]
        # x is [B, N, C]
        pooled_x = ops.matmul(ops.swapaxes(S, 1, 2), x)  # [B, K, C]
        x_out = self.lin(pooled_x)

        return x_out, S

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"num_clusters={self.num_clusters})"
        )
