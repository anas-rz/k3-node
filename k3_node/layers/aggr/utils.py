from typing import Optional
from keras import initializers, layers, ops


class MultiheadAttentionBlock(layers.Layer):
    r"""The Multihead Attention Block (MAB) from the `"Set Transformer: A
    Framework for Attention-based Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.heads = heads
        self.dropout = dropout

        key_dim = max(channels // heads, 1)
        self.attn = layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=key_dim,
            dropout=dropout,
        )
        self.lin = layers.Dense(channels)
        self.layer_norm1 = layers.LayerNormalization() if layer_norm else None
        self.layer_norm2 = layers.LayerNormalization() if layer_norm else None

    def reset_parameters(self):
        pass

    def call(
        self,
        x,
        y,
        x_mask: Optional[any] = None,
        y_mask: Optional[any] = None,
        training: bool = False,
    ):
        attn_mask = None
        if y_mask is not None:
            # y_mask shape: [B, S], expand to [B, 1, S]
            attn_mask = ops.expand_dims(ops.cast(y_mask, "bool"), axis=1)

        out = self.attn(
            query=x,
            value=y,
            key=y,
            attention_mask=attn_mask,
            training=training,
        )

        if x_mask is not None:
            out = out * ops.cast(ops.expand_dims(x_mask, axis=-1), out.dtype)

        out = out + x
        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = out + ops.relu(self.lin(out))
        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"heads={self.heads}, layer_norm={self.layer_norm1 is not None}, "
            f"dropout={self.dropout})"
        )


class SetAttentionBlock(layers.Layer):
    r"""The Set Attention Block (SAB) from the `"Set Transformer"` paper."""

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def call(self, x, mask: Optional[any] = None, training: bool = False):
        return self.mab(x, x, x_mask=mask, y_mask=mask, training=training)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.mab.channels}, "
            f"heads={self.mab.heads}, layer_norm={self.mab.layer_norm1 is not None}, "
            f"dropout={self.mab.dropout})"
        )


class InducedSetAttentionBlock(layers.Layer):
    r"""The Induced Set Attention Block (ISAB) from the `"Set Transformer"` paper."""

    def __init__(
        self,
        channels: int,
        num_induced_points: int,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_induced_points = num_induced_points

        self.ind = self.add_weight(
            shape=(1, num_induced_points, channels),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            name="ind",
        )
        self.mab1 = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)
        self.mab2 = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def reset_parameters(self):
        init = initializers.GlorotUniform()
        self.ind.assign(init(self.ind.shape, dtype=self.ind.dtype))
        self.mab1.reset_parameters()
        self.mab2.reset_parameters()

    def call(self, x, mask: Optional[any] = None, training: bool = False):
        B = ops.shape(x)[0]
        ind_expanded = ops.broadcast_to(self.ind, (B, self.num_induced_points, self.channels))
        h = self.mab1(ind_expanded, x, y_mask=mask, training=training)
        return self.mab2(x, h, x_mask=mask, training=training)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"num_induced_points={self.num_induced_points}, "
            f"heads={self.mab1.heads})"
        )


class PoolingByMultiheadAttention(layers.Layer):
    r"""The Pooling by Multihead Attention (PMA) layer from the `"Set Transformer"` paper."""

    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        heads: int = 1,
        layer_norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_seed_points = num_seed_points
        self.lin = layers.Dense(channels)
        self.seed = self.add_weight(
            shape=(1, num_seed_points, channels),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            name="seed",
        )
        self.mab = MultiheadAttentionBlock(channels, heads, layer_norm, dropout)

    def reset_parameters(self):
        self.lin.reset_parameters()
        init = initializers.GlorotUniform()
        self.seed.assign(init(self.seed.shape, dtype=self.seed.dtype))
        self.mab.reset_parameters()

    def call(self, x, mask: Optional[any] = None, training: bool = False):
        B = ops.shape(x)[0]
        x_proj = ops.relu(self.lin(x))
        seed_expanded = ops.broadcast_to(self.seed, (B, self.num_seed_points, self.channels))
        return self.mab(seed_expanded, x_proj, y_mask=mask, training=training)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"num_seed_points={self.num_seed_points}, "
            f"heads={self.mab.heads})"
        )

