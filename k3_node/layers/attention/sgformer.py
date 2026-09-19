from typing import Optional
import keras
from keras import ops


class SGFormerAttention(keras.layers.Layer):
    r"""The simple global attention mechanism from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
            (default: :obj:`1`)
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: Optional[int] = 64,
        qkv_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.channels = channels
        self.heads = heads
        self.head_channels = head_channels
        self.qkv_bias = qkv_bias

        inner_channels = head_channels * heads
        self.q = keras.layers.Dense(inner_channels, use_bias=qkv_bias)
        self.k = keras.layers.Dense(inner_channels, use_bias=qkv_bias)
        self.v = keras.layers.Dense(inner_channels, use_bias=qkv_bias)

    def call(self, x, mask: Optional[any] = None):
        shape = ops.shape(x)
        B = shape[0]
        N = shape[1]

        qs = self.q(x)
        ks = self.k(x)
        vs = self.v(x)

        qs = ops.reshape(qs, (B, N, self.heads, self.head_channels))
        ks = ops.reshape(ks, (B, N, self.heads, self.head_channels))
        vs = ops.reshape(vs, (B, N, self.heads, self.head_channels))

        if mask is not None:
            m = ops.expand_dims(ops.expand_dims(mask, -1), -1)
            m = ops.cast(m, "bool")
            vs = ops.where(m, vs, ops.zeros_like(vs))

        epsilon = 1e-6
        qs = ops.where(ops.equal(qs, 0.0), epsilon, qs)
        ks = ops.where(ops.equal(ks, 0.0), epsilon, ks)

        qs_norm = ops.norm(qs, axis=-1, keepdims=True)
        ks_norm = ops.norm(ks, axis=-1, keepdims=True)
        qs = qs / ops.maximum(qs_norm, 1e-12)
        ks = ks / ops.maximum(ks_norm, 1e-12)

        # numerator
        kvs = ops.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = ops.einsum("bnhm,bhmd->bnhd", qs, kvs)
        n_float = ops.cast(N, vs.dtype)
        attention_num = attention_num + n_float * vs

        # denominator
        all_ones = ops.ones((B, N), dtype=ks.dtype)
        ks_sum = ops.einsum("blhm,bl->bhm", ks, all_ones)
        attention_normalizer = ops.einsum("bnhm,bhm->bnh", qs, ks_sum)
        attention_normalizer = ops.expand_dims(attention_normalizer, -1)
        attention_normalizer = attention_normalizer + n_float
        attn_output = attention_num / attention_normalizer

        return ops.mean(attn_output, axis=2)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"heads={self.heads}, "
                f"head_channels={self.head_channels})")

