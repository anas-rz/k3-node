from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import keras
from keras import ops


def to_dense_batch(x, batch=None):
    if batch is None:
        mask = ops.ones((1, ops.shape(x)[0]), dtype="bool")
        return ops.expand_dims(x, axis=0), mask

    batch_np = ops.convert_to_numpy(batch).astype(np.int64)
    N = len(batch_np)
    B = int(np.max(batch_np)) + 1 if N > 0 else 1
    counts = np.bincount(batch_np, minlength=B)
    max_nodes = int(np.max(counts)) if len(counts) > 0 else 0

    x_np = ops.convert_to_numpy(x)
    C = x_np.shape[-1]
    dense_out = np.zeros((B, max_nodes, C), dtype=np.float32)
    mask = np.zeros((B, max_nodes), dtype=bool)

    offsets = np.zeros(B, dtype=np.int64)
    for b, feat in zip(batch_np, x_np):
        off = offsets[b]
        dense_out[b, off] = feat
        mask[b, off] = True
        offsets[b] += 1

    return ops.convert_to_tensor(dense_out, dtype=x.dtype), ops.convert_to_tensor(mask)


class GPSConv(keras.layers.Layer):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    Args:
        channels (int): Size of each input sample.
        conv (keras.layers.Layer, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions. (default: :obj:`1`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
        act (str, optional): Activation function. (default: :obj:`"relu"`)
        norm (str, optional): Normalization function. (default: :obj:`"batch_norm"`)
    """

    def __init__(
        self,
        channels: int,
        conv: Optional[keras.layers.Layer] = None,
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        norm: Optional[str] = "batch_norm",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout_rate = dropout
        self.act = act
        self.norm_name = norm

        self.attn = keras.layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=channels // heads,
            value_dim=channels // heads,
            output_shape=channels,
        )

        self.mlp_l1 = keras.layers.Dense(channels * 2)
        self.mlp_l2 = keras.layers.Dense(channels)

        if norm == "batch_norm":
            self.norm1 = keras.layers.BatchNormalization(axis=-1) if conv is not None else None
            self.norm2 = keras.layers.BatchNormalization(axis=-1)
            self.norm3 = keras.layers.BatchNormalization(axis=-1)
        elif norm == "layer_norm":
            self.norm1 = keras.layers.LayerNormalization(axis=-1) if conv is not None else None
            self.norm2 = keras.layers.LayerNormalization(axis=-1)
            self.norm3 = keras.layers.LayerNormalization(axis=-1)
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None

    def build(self, input_shape=None):
        if self.conv is not None and hasattr(self.conv, "build") and not self.conv.built:
            self.conv.build(input_shape)
        if not self.attn.built:
            self.attn.build((None, None, self.channels), (None, None, self.channels))
        if not self.mlp_l1.built:
            self.mlp_l1.build((None, self.channels))
        if not self.mlp_l2.built:
            self.mlp_l2.build((None, self.channels * 2))
        super().build(input_shape)

    def call(self, x, edge_index, batch=None, **kwargs):
        if not self.built:
            self.build((None, self.channels))

        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = h + x
            if self.norm1 is not None:
                h = self.norm1(h)
            hs.append(h)

        # Global attention
        h_dense, mask = to_dense_batch(x, batch)
        # Attention mask for Keras: shape (B, 1, max_nodes)
        attn_mask = ops.expand_dims(mask, axis=1)
        attn_out = self.attn(h_dense, h_dense, attention_mask=attn_mask)

        # Unpack dense batch to original flat shape
        if batch is None:
            h_global = attn_out[0]
        else:
            mask_flat = ops.reshape(mask, (-1,))
            out_flat = ops.reshape(attn_out, (-1, self.channels))
            idx = ops.where(mask_flat)[0]
            h_global = ops.take(out_flat, idx, axis=0)

        h_global = h_global + x
        if self.norm2 is not None:
            h_global = self.norm2(h_global)
        hs.append(h_global)

        # Combine local and global
        if len(hs) > 1:
            out = hs[0] + hs[1]
        else:
            out = hs[0]

        # MLP
        mlp_h = ops.relu(self.mlp_l1(out))
        mlp_out = self.mlp_l2(mlp_h)
        out = out + mlp_out

        if self.norm3 is not None:
            out = self.norm3(out)

        return out
