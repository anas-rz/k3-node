import keras
from keras import ops


class PolynormerAttention(keras.layers.Layer):
    def __init__(
        self,
        channels,
        heads,
        head_channels=64,
        beta=0.9,
        qkv_bias=False,
        qk_shared=True,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.channels = channels
        self.heads = heads
        self.head_channels = head_channels
        self.beta = beta
        self.qk_shared = qk_shared

        inner_channels = heads * head_channels

        self.h_lins = keras.layers.Dense(inner_channels)

        if not qk_shared:
            self.q = keras.layers.Dense(
                inner_channels,
                use_bias=qkv_bias,
            )

        self.k = keras.layers.Dense(
            inner_channels,
            use_bias=qkv_bias,
        )

        self.v = keras.layers.Dense(
            inner_channels,
            use_bias=qkv_bias,
        )

        self.lns = keras.layers.LayerNormalization(epsilon=1e-5)
        self.lin_out = keras.layers.Dense(inner_channels)
        self.dropout = keras.layers.Dropout(dropout)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None, training=None):
        shape = ops.shape(x)

        B = shape[0]
        N = shape[1]

        h = self.h_lins(x)

        k = self.k(x)
        k = ops.sigmoid(k)
        k = ops.reshape(
            k,
            (B, N, self.head_channels, self.heads),
        )

        if self.qk_shared:
            q = k
        else:
            q = ops.sigmoid(self.q(x))
            q = ops.reshape(
                q,
                (B, N, self.head_channels, self.heads),
            )

        v = self.v(x)
        v = ops.reshape(
            v,
            (B, N, self.head_channels, self.heads),
        )

        if mask is not None:
            mask = ops.expand_dims(mask, axis=-1)
            mask = ops.expand_dims(mask, axis=-1)
            v = v * ops.cast(mask, v.dtype)

        # numerator
        kv = ops.einsum(
            "bndh,bnmh->bdmh",
            k,
            v,
        )

        num = ops.einsum(
            "bndh,bdmh->bnmh",
            q,
            kv,
        )

        # denominator
        k_sum = ops.einsum(
            "bndh->bdh",
            k,
        )

        den = ops.einsum(
            "bndh,bdh->bnh",
            q,
            k_sum,
        )

        den = ops.expand_dims(den, axis=2)

        x = num / (den + 1e-6)

        x = ops.reshape(
            x,
            (B, N, self.head_channels * self.heads),
        )

        x = self.lns(x)
        x = x * (h + self.beta)

        x = self.lin_out(x)
        x = ops.relu(x)

        x = self.dropout(
            x,
            training=training,
        )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "heads": self.heads,
                "head_channels": self.head_channels,
                "beta": self.beta,
                "qk_shared": self.qk_shared,
            }
        )
        return config