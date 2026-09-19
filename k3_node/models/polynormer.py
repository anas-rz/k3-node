from typing import Optional

import keras
from keras import ops
import numpy as np

from k3_node.layers.conv import GATConv, GCNConv
from k3_node.layers.attention import PolynormerAttention


def _to_dense_batch_sorted(x, batch):
    """Convert sparse node features to dense [B, N_max, F] representation.

    Assumes batch is sorted (ascending). Returns (dense_x, mask).
    Uses ops.scatter_update for full multi-backend compatibility.
    """
    batch_np = ops.convert_to_numpy(batch).astype(np.int64)
    N = len(batch_np)
    B = int(np.max(batch_np)) + 1 if N > 0 else 1
    counts = np.bincount(batch_np, minlength=B)
    max_nodes = int(np.max(counts)) if len(counts) > 0 else 0

    x_np = ops.convert_to_numpy(x)
    F = x_np.shape[-1]

    dense_np = np.zeros((B, max_nodes, F), dtype=np.float32)
    mask_np = np.zeros((B, max_nodes), dtype=bool)
    offsets = np.zeros(B, dtype=np.int64)
    for i, b in enumerate(batch_np):
        off = offsets[b]
        dense_np[b, off] = x_np[i]
        mask_np[b, off] = True
        offsets[b] += 1

    return (
        ops.convert_to_tensor(dense_np, dtype=x.dtype),
        ops.convert_to_tensor(mask_np, dtype="bool"),
    )


class Polynormer(keras.layers.Layer):
    r"""The Polynormer module from the `"Polynormer: polynomial-expressive
    graph transformer in linear time"
    <https://arxiv.org/abs/2403.01232>`_ paper.

    Args:
        in_channels (int): Input channels.
        hidden_channels (int): Hidden channels.
        out_channels (int): Output channels.
        local_layers (int): The number of local attention layers.
            (default: :obj:`7`)
        global_layers (int): The number of global attention layers.
            (default: :obj:`2`)
        in_dropout (float): Input dropout rate.
            (default: :obj:`0.15`)
        dropout (float): Dropout rate.
            (default: :obj:`0.5`)
        global_dropout (float): Global dropout rate.
            (default: :obj:`0.5`)
        heads (int): The number of heads.
            (default: :obj:`1`)
        beta (float): Aggregate type.
            (default: :obj:`0.9`)
        qk_shared (bool, optional): Whether weight of query and key are shared.
            (default: :obj:`True`)
        pre_ln (bool): Pre layer normalization.
            (default: :obj:`False`)
        post_bn (bool): Post batch normalization.
            (default: :obj:`True`)
        local_attn (bool): Whether use local attention (GATConv vs GCNConv).
            (default: :obj:`False`)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        local_layers: int = 7,
        global_layers: int = 2,
        in_dropout: float = 0.15,
        dropout: float = 0.5,
        global_dropout: float = 0.5,
        heads: int = 1,
        beta: float = 0.9,
        qk_shared: bool = False,
        pre_ln: bool = False,
        post_bn: bool = True,
        local_attn: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._global = False
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn
        self.beta = beta
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.local_attn = local_attn

        inner_channels = heads * hidden_channels

        self.h_lins = []
        self.local_convs = []
        self.lins = []
        self.lns = []
        self.pre_lns = [] if pre_ln else None
        self.post_bns = [] if post_bn else None

        # ---- First local layer ----
        self.h_lins.append(keras.layers.Dense(inner_channels))
        if local_attn:
            self.local_convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                        add_self_loops=False, bias=False)
            )
        else:
            self.local_convs.append(
                GCNConv(in_channels, inner_channels, cached=False, normalize=True)
            )
        self.lins.append(keras.layers.Dense(inner_channels))
        self.lns.append(keras.layers.LayerNormalization(epsilon=1e-5))
        if pre_ln:
            self.pre_lns.append(keras.layers.LayerNormalization(epsilon=1e-5))
        if post_bn:
            self.post_bns.append(
                keras.layers.BatchNormalization(
                    center=True, scale=True, momentum=0.9, epsilon=1e-5,
                )
            )

        # ---- Subsequent local layers ----
        for _ in range(local_layers - 1):
            self.h_lins.append(keras.layers.Dense(inner_channels))
            if local_attn:
                self.local_convs.append(
                    GATConv(inner_channels, hidden_channels, heads=heads,
                            concat=True, add_self_loops=False, bias=False)
                )
            else:
                self.local_convs.append(
                    GCNConv(inner_channels, inner_channels, cached=False,
                            normalize=True)
                )
            self.lins.append(keras.layers.Dense(inner_channels))
            self.lns.append(keras.layers.LayerNormalization(epsilon=1e-5))
            if pre_ln:
                self.pre_lns.append(keras.layers.LayerNormalization(epsilon=1e-5))
            if post_bn:
                self.post_bns.append(
                    keras.layers.BatchNormalization(
                        center=True, scale=True, momentum=0.9, epsilon=1e-5,
                    )
                )

        self.lin_in = keras.layers.Dense(inner_channels)
        self.ln = keras.layers.LayerNormalization(epsilon=1e-5)

        self.global_attn = [
            PolynormerAttention(
                channels=hidden_channels,
                heads=heads,
                head_channels=hidden_channels,
                beta=beta,
                dropout=global_dropout,
                qk_shared=qk_shared,
            )
            for _ in range(global_layers)
        ]

        self.pred_local = keras.layers.Dense(out_channels)
        self.pred_global = keras.layers.Dense(out_channels)

        self._in_dropout = keras.layers.Dropout(in_dropout)
        self._dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        # Keras layers reinitialize on next forward; no-op for unbuilt layers.
        pass

    def call(self, x, edge_index, batch: Optional[object] = None, training=None):
        r"""Forward pass.

        Args:
            x (Tensor): The input node features.
            edge_index (Tensor): The edge indices.
            batch (Tensor, optional): The batch vector assigning each node to
                a graph. (default: :obj:`None`)
            training (bool, optional): Whether in training mode.
                (default: :obj:`None`)
        """
        x = self._in_dropout(x, training=training)

        # ---- Equivariant local attention ----
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = ops.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x, training=training)
            x = ops.relu(x)
            x = self._dropout(x, training=training)
            x = (1 - self.beta) * self.lns[i](h * x) + self.beta * x
            x_local = x_local + x

        # ---- Equivariant global attention ----
        if self._global:
            # Sort nodes by batch assignment (required by to_dense_batch)
            batch_np = ops.convert_to_numpy(batch).astype(np.int64)
            indices = np.argsort(batch_np, kind='stable')
            rev_perm = np.empty_like(indices)
            rev_perm[indices] = np.arange(len(indices))

            batch_sorted = ops.convert_to_tensor(batch_np[indices], dtype="int32")
            x_local_sorted = ops.take(x_local, ops.convert_to_tensor(indices, dtype="int32"), axis=0)
            x_local_sorted = self.ln(x_local_sorted)

            x_global, mask = _to_dense_batch_sorted(x_local_sorted, batch_sorted)
            for attn in self.global_attn:
                x_global = attn(x_global, mask=mask, training=training)

            # Flatten and undo sort
            x_global_flat = x_global[mask]  # [N, F]
            x = ops.take(x_global_flat, ops.convert_to_tensor(rev_perm, dtype="int32"), axis=0)
            x = self.pred_global(x)
        else:
            x = self.pred_local(x_local)

        return ops.log_softmax(x, axis=-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.hidden_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'heads={self.heads})')

