import numpy as np
import pytest
import keras
from keras import ops

from k3_node.layers.attention import (
    SelfMultiheadAttentionWithPair,
    TransformerEncoderLayerWithPair,
    TriangleMultiplication,
    OuterProduct,
)


def test_self_multihead_attention_with_pair():
    bsz = 2
    seq_len = 6
    embed_dim = 32
    num_heads = 4

    x = ops.convert_to_tensor(np.random.randn(bsz, seq_len, embed_dim).astype("float32"))
    attn_bias = ops.convert_to_tensor(np.random.randn(bsz, num_heads, seq_len, seq_len).astype("float32"))
    padding_mask = ops.convert_to_tensor(np.zeros((bsz, seq_len), dtype="bool"))

    layer = SelfMultiheadAttentionWithPair(embed_dim=embed_dim, num_heads=num_heads)
    out = layer(x, key_padding_mask=padding_mask, attn_bias=attn_bias)
    assert ops.shape(out) == (bsz, seq_len, embed_dim)

    # Test return_attn=True
    out, weights, probs = layer(x, key_padding_mask=padding_mask, attn_bias=attn_bias, return_attn=True)
    assert ops.shape(out) == (bsz, seq_len, embed_dim)
    assert ops.shape(weights) == (bsz, num_heads, seq_len, seq_len)
    assert ops.shape(probs) == (bsz, num_heads, seq_len, seq_len)


def test_transformer_encoder_layer_with_pair():
    bsz = 2
    seq_len = 6
    embed_dim = 32
    num_heads = 4

    x = ops.convert_to_tensor(np.random.randn(bsz, seq_len, embed_dim).astype("float32"))
    attn_bias = ops.convert_to_tensor(np.random.randn(bsz, num_heads, seq_len, seq_len).astype("float32"))

    layer = TransformerEncoderLayerWithPair(
        embed_dim=embed_dim,
        ffn_embed_dim=64,
        attention_heads=num_heads,
        post_ln=False,
    )
    out = layer(x, attn_bias=attn_bias)
    assert ops.shape(out) == (bsz, seq_len, embed_dim)

    # Post-LN
    layer_post = TransformerEncoderLayerWithPair(
        embed_dim=embed_dim,
        ffn_embed_dim=64,
        attention_heads=num_heads,
        post_ln=True,
    )
    out_post = layer_post(x, attn_bias=attn_bias)
    assert ops.shape(out_post) == (bsz, seq_len, embed_dim)


def test_triangle_multiplication():
    bsz = 2
    seq_len = 5
    pair_dim = 16
    hidden_dim = 8

    pair = ops.convert_to_tensor(np.random.randn(bsz, seq_len, seq_len, pair_dim).astype("float32"))

    tri_out = TriangleMultiplication(pair_dim=pair_dim, hidden_dim=hidden_dim, mode="outgoing")
    out = tri_out(pair)
    assert ops.shape(out) == (bsz, seq_len, seq_len, pair_dim)

    tri_in = TriangleMultiplication(pair_dim=pair_dim, hidden_dim=hidden_dim, mode="incoming")
    out = tri_in(pair)
    assert ops.shape(out) == (bsz, seq_len, seq_len, pair_dim)


def test_outer_product():
    bsz = 2
    seq_len = 6
    embed_dim = 32
    pair_dim = 16

    x = ops.convert_to_tensor(np.random.randn(bsz, seq_len, embed_dim).astype("float32"))
    op = OuterProduct(embed_dim=embed_dim, pair_dim=pair_dim, hidden_dim=8)
    out = op(x)
    assert ops.shape(out) == (bsz, seq_len, seq_len, pair_dim)

