import math
from typing import Optional, Union, Tuple, Callable

import keras
from keras import layers, ops


def _get_activation(activation_fn: Optional[Union[str, Callable]]):
    if activation_fn is None:
        return None
    if isinstance(activation_fn, str):
        fn = activation_fn.lower()
        if fn == "gelu":
            return layers.Activation("gelu")
        elif fn == "relu":
            return layers.ReLU()
        elif fn == "tanh":
            return layers.Activation("tanh")
        elif fn == "silu" or fn == "swish":
            return layers.Activation("silu")
        elif fn == "linear":
            return layers.Activation("linear")
        else:
            return layers.Activation(activation_fn)
    elif isinstance(activation_fn, layers.Layer):
        return activation_fn
    elif callable(activation_fn):
        return layers.Activation(activation_fn)
    return None


class SelfMultiheadAttentionWithPair(layers.Layer):
    r"""Multihead self-attention layer supporting additive pair-level attention bias.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        dropout (float, optional): Dropout probability for attention weights. (default: ``0.1``)
        bias (bool, optional): Whether to include bias terms in linear projections. (default: ``True``)
        scaling_factor (float, optional): Scaling factor multiplier for attention keys. (default: ``1.0``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        scaling_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bias = bias
        self.scaling_factor = float(scaling_factor)

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = embed_dim // num_heads
        self.scaling = (self.head_dim * self.scaling_factor) ** -0.5

        self.in_proj = layers.Dense(embed_dim * 3, use_bias=bias, name="in_proj")
        self.out_proj = layers.Dense(embed_dim, use_bias=bias, name="out_proj")
        self.attn_dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def build(self, input_shape=None):
        if not self.built:
            self.in_proj.build((None, None, self.embed_dim))
            self.out_proj.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(
        self,
        query,
        key_padding_mask=None,
        attn_bias=None,
        return_attn: bool = False,
        training: bool = False,
    ):
        r"""
        Args:
            query (Tensor): Input tensor of shape ``[batch_size, seq_len, embed_dim]``.
            key_padding_mask (Tensor, optional): Mask indicating padding tokens of shape
                ``[batch_size, seq_len]`` (True/1 for padding, False/0 for valid tokens).
            attn_bias (Tensor, optional): Additive pairwise attention bias of shape
                ``[batch_size, num_heads, seq_len, seq_len]`` or ``[batch_size * num_heads, seq_len, seq_len]``.
            return_attn (bool, optional): Whether to return raw and post-softmax attention weights.
            training (bool, optional): Whether the layer is in training mode.

        Returns:
            Tensor or Tuple[Tensor, Tensor, Tensor]: Attention output and optionally attention weights.
        """
        shape = ops.shape(query)
        bsz = shape[0]
        tgt_len = shape[1]

        qkv = self.in_proj(query)  # [bsz, tgt_len, embed_dim * 3]
        q, k, v = ops.split(qkv, 3, axis=-1)

        # Reshape to [bsz, num_heads, seq_len, head_dim]
        q = ops.reshape(q, (bsz, tgt_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        q = q * self.scaling

        k = ops.reshape(k, (bsz, tgt_len, self.num_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))

        v = ops.reshape(v, (bsz, tgt_len, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        # [bsz, num_heads, tgt_len, tgt_len]
        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        if attn_bias is not None:
            bias_shape = ops.shape(attn_bias)
            if len(bias_shape) == 3:
                # [bsz * num_heads, tgt_len, tgt_len] -> [bsz, num_heads, tgt_len, tgt_len]
                attn_bias = ops.reshape(attn_bias, (bsz, self.num_heads, tgt_len, tgt_len))
            elif len(bias_shape) == 4 and bias_shape[-1] == self.num_heads:
                # [bsz, tgt_len, tgt_len, num_heads] -> [bsz, num_heads, tgt_len, tgt_len]
                attn_bias = ops.transpose(attn_bias, (0, 3, 1, 2))
            attn_weights = attn_weights + attn_bias

        if key_padding_mask is not None:
            # key_padding_mask: True or 1 for padding
            mask = ops.cast(key_padding_mask, "bool")
            mask = ops.expand_dims(ops.expand_dims(mask, axis=1), axis=2)  # [bsz, 1, 1, tgt_len]
            attn_weights = ops.where(mask, ops.cast(-1e9, attn_weights.dtype), attn_weights)

        attn_probs = ops.softmax(attn_weights, axis=-1)
        if self.attn_dropout is not None:
            attn_probs = self.attn_dropout(attn_probs, training=training)

        o = ops.matmul(attn_probs, v)  # [bsz, num_heads, tgt_len, head_dim]
        o = ops.transpose(o, (0, 2, 1, 3))  # [bsz, tgt_len, num_heads, head_dim]
        o = ops.reshape(o, (bsz, tgt_len, self.embed_dim))
        o = self.out_proj(o)

        if not return_attn:
            return o
        return o, attn_weights, attn_probs


class TransformerEncoderLayerWithPair(layers.Layer):
    r"""Transformer Encoder Layer with pair representation bias and update.

    Args:
        embed_dim (int, optional): Node feature dimension. (default: ``768``)
        ffn_embed_dim (int, optional): Feed-forward network hidden dimension. (default: ``3072``)
        attention_heads (int, optional): Number of attention heads. (default: ``8``)
        dropout (float, optional): Dropout probability. (default: ``0.1``)
        attention_dropout (float, optional): Attention dropout probability. (default: ``0.1``)
        activation_dropout (float, optional): Activation dropout probability in FFN. (default: ``0.0``)
        activation_fn (str or Callable, optional): Non-linear activation function. (default: ``"gelu"``)
        post_ln (bool, optional): Whether to use Post-LN instead of Pre-LN. (default: ``False``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: Union[str, Callable] = "gelu",
        post_ln: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout_rate = dropout
        self.attention_dropout_rate = attention_dropout
        self.activation_dropout_rate = activation_dropout
        self.activation_fn_name = activation_fn
        self.post_ln = post_ln

        self.self_attn = SelfMultiheadAttentionWithPair(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            name="self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="self_attn_layer_norm")

        self.fc1 = layers.Dense(ffn_embed_dim, name="fc1")
        self.activation_fn = _get_activation(activation_fn)
        self.fc2 = layers.Dense(embed_dim, name="fc2")
        self.final_layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="final_layer_norm")

        self.dropout1 = layers.Dropout(dropout) if dropout > 0.0 else None
        self.dropout2 = layers.Dropout(dropout) if dropout > 0.0 else None
        self.act_dropout = layers.Dropout(activation_dropout) if activation_dropout > 0.0 else None

    def build(self, input_shape=None):
        if not self.built:
            self.self_attn.build((None, None, self.embed_dim))
            self.self_attn_layer_norm.build((None, None, self.embed_dim))
            self.fc1.build((None, None, self.embed_dim))
            self.fc2.build((None, None, self.ffn_embed_dim))
            self.final_layer_norm.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(
        self,
        x,
        attn_bias=None,
        padding_mask=None,
        return_attn: bool = False,
        training: bool = False,
    ):
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)

        attn_out = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
            training=training,
        )

        attn_weights = None
        attn_probs = None
        if return_attn:
            x, attn_weights, attn_probs = attn_out
        else:
            x = attn_out

        if self.dropout1 is not None:
            x = self.dropout1(x, training=training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        if self.act_dropout is not None:
            x = self.act_dropout(x, training=training)
        x = self.fc2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x, training=training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)

        if not return_attn:
            return x
        return x, attn_weights, attn_probs


class TriangleMultiplication(layers.Layer):
    r"""Triangle Multiplicative Update layer (AlphaFold2 / Uni-Mol2 / Uni-Mol+).

    Args:
        pair_dim (int): Pair feature dimension.
        hidden_dim (int): Intermediate channel dimension.
        mode (str, optional): Either ``"outgoing"`` or ``"incoming"``. (default: ``"outgoing"``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        pair_dim: int,
        hidden_dim: int,
        mode: str = "outgoing",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("outgoing", "incoming"):
            raise ValueError(f"Unknown TriangleMultiplication mode '{mode}', must be 'outgoing' or 'incoming'")
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm")
        self.proj_a = layers.Dense(hidden_dim, use_bias=False, name="proj_a")
        self.proj_b = layers.Dense(hidden_dim, use_bias=False, name="proj_b")
        self.gate_a = layers.Dense(hidden_dim, use_bias=True, name="gate_a")
        self.gate_b = layers.Dense(hidden_dim, use_bias=True, name="gate_b")

        self.gate_out = layers.Dense(pair_dim, use_bias=True, name="gate_out")
        self.proj_out = layers.Dense(pair_dim, use_bias=True, name="proj_out")
        self.norm_out = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm_out")

    def build(self, input_shape=None):
        if not self.built:
            self.norm.build((None, None, None, self.pair_dim))
            self.proj_a.build((None, None, None, self.pair_dim))
            self.proj_b.build((None, None, None, self.pair_dim))
            self.gate_a.build((None, None, None, self.pair_dim))
            self.gate_b.build((None, None, None, self.pair_dim))
            self.gate_out.build((None, None, None, self.pair_dim))
            self.proj_out.build((None, None, None, self.hidden_dim))
            self.norm_out.build((None, None, None, self.hidden_dim))
        super().build(input_shape)

    def call(self, pair, mask=None, training: bool = False):
        r"""
        Args:
            pair (Tensor): Pair tensor of shape ``[batch_size, seq_len, seq_len, pair_dim]``.
            mask (Tensor, optional): Optional pair mask of shape ``[batch_size, seq_len, seq_len]``.
            training (bool, optional): Training flag.

        Returns:
            Tensor: Updated pair tensor with residual addition.
        """
        residual = pair
        x = self.norm(pair)

        a = self.proj_a(x) * ops.sigmoid(self.gate_a(x))
        b = self.proj_b(x) * ops.sigmoid(self.gate_b(x))

        if mask is not None:
            m = ops.expand_dims(ops.cast(mask, a.dtype), axis=-1)
            a = a * m
            b = b * m

        # Multiplicative update
        if self.mode == "outgoing":
            # [B, N, N, C] = \sum_k a[i, k] * b[j, k]
            # transpose b to [B, K, J, C] for matrix multiply along K
            # Using ops.einsum for standard formulation:
            out = ops.einsum("bikc,bjkc->bijc", a, b)
        else:
            # incoming: \sum_k a[k, i] * b[k, j]
            out = ops.einsum("bkic,bkjc->bijc", a, b)

        out = self.norm_out(out)
        out = self.proj_out(out) * ops.sigmoid(self.gate_out(pair))
        return residual + out


class OuterProduct(layers.Layer):
    r"""Outer product mean layer updating pair representations from atom representations.

    Args:
        embed_dim (int): Node/atom embedding dimension.
        pair_dim (int): Pair representation dimension.
        hidden_dim (int, optional): Intermediate projection dimension. (default: ``32``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        pair_dim: int,
        hidden_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim

        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm")
        self.proj_left = layers.Dense(hidden_dim, use_bias=True, name="proj_left")
        self.proj_right = layers.Dense(hidden_dim, use_bias=True, name="proj_right")
        self.proj_out = layers.Dense(pair_dim, use_bias=True, name="proj_out")

    def build(self, input_shape=None):
        if not self.built:
            self.norm.build((None, None, self.embed_dim))
            self.proj_left.build((None, None, self.embed_dim))
            self.proj_right.build((None, None, self.embed_dim))
            self.proj_out.build((None, None, None, self.hidden_dim))
        super().build(input_shape)

    def call(self, x, mask=None, training: bool = False):
        r"""
        Args:
            x (Tensor): Node/atom representations of shape ``[batch_size, seq_len, embed_dim]``.
            mask (Tensor, optional): Node mask of shape ``[batch_size, seq_len]``.
            training (bool, optional): Training flag.

        Returns:
            Tensor: Pair update of shape ``[batch_size, seq_len, seq_len, pair_dim]``.
        """
        x_norm = self.norm(x)
        if mask is not None:
            m = ops.expand_dims(ops.cast(mask, x.dtype), axis=-1)
            x_norm = x_norm * m

        left = self.proj_left(x_norm)    # [B, N, C]
        right = self.proj_right(x_norm)  # [B, N, C]

        # [B, N, 1, C] * [B, 1, N, C] -> [B, N, N, C]
        prod = ops.expand_dims(left, axis=2) * ops.expand_dims(right, axis=1)
        return self.proj_out(prod)

