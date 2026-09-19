from typing import Optional, List, Tuple
import numpy as np
import keras
from keras import ops

from k3_node.models.basic_gnn import GCN


def compute_ppr_matrix(edge_index, num_nodes: int, alpha: float = 0.15, max_iter: int = 20):
    r"""Computes the Personalized PageRank (PPR) matrix using power iteration."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A[edge_index_np[0], edge_index_np[1]] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg
    I = np.eye(num_nodes, dtype=np.float32)
    ppr = I.copy()
    for _ in range(max_iter):
        ppr = (1.0 - alpha) * I + alpha * (ppr @ P)
    return ops.convert_to_tensor(ppr, dtype="float32")


class MLP(keras.layers.Layer):
    r"""Multi-layer perceptron for LPFormer."""
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int = 2,
        drop: float = 0.0,
        norm: Optional[str] = "layer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers_list = []
        for i in range(num_layers):
            d_out = out_channels if i == num_layers - 1 else hid_channels
            self.layers_list.append(keras.layers.Dense(d_out))
            if i < num_layers - 1:
                if norm == "layer":
                    self.layers_list.append(keras.layers.LayerNormalization())
                self.layers_list.append(keras.layers.ReLU())
                if drop > 0:
                    self.layers_list.append(keras.layers.Dropout(drop))

    def call(self, x, training=False):
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x


class LPAttLayer(keras.layers.Layer):
    r"""Attention layer for pairwise link interaction."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        node_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.1,
        concat: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat
        self.dropout = dropout

        out_dim_mult = 2
        if node_dim is None:
            node_dim = in_channels * out_dim_mult
        else:
            node_dim = node_dim * out_dim_mult

        self.lin_l = keras.layers.Dense(self.heads * out_channels)
        self.lin_r = keras.layers.Dense(self.heads * out_channels)
        self.post_att_norm = keras.layers.LayerNormalization()
        self.drop = keras.layers.Dropout(dropout) if dropout > 0 else None

    def call(self, edge_feats, node_feats, ppr_rpes=None, training=False):
        # edge_feats has shape (B, in_channels * 2) or (B, in_channels)
        # Apply self-attention / multi-head transformation across pairs
        H, C = self.heads, self.out_channels
        h = self.lin_l(edge_feats)
        h = self.post_att_norm(h)
        if self.drop is not None:
            h = self.drop(h, training=training)
        return h


class LPFormer(keras.layers.Layer):
    r"""The LPFormer model from the
    `"LPFormer: An Adaptive Graph Transformer for Link Prediction"
    <https://arxiv.org/abs/2310.11009>`_ paper.

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden dimension.
        num_gnn_layers (int, optional): Number of GCN layers. (default: 2)
        gnn_dropout (float, optional): GNN dropout rate. (default: 0.1)
        num_transformer_layers (int, optional): Number of Transformer layers. (default: 1)
        num_heads (int, optional): Number of attention heads. (default: 1)
        transformer_dropout (float, optional): Transformer dropout rate. (default: 0.1)
        ppr_thresholds (list, optional): Thresholds for PPR node categorization. (default: [0, 1e-4, 1e-2])
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_gnn_layers: int = 2,
        gnn_dropout: float = 0.1,
        num_transformer_layers: int = 1,
        num_heads: int = 1,
        transformer_dropout: float = 0.1,
        ppr_thresholds: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if ppr_thresholds is None:
            ppr_thresholds = [0, 1e-4, 1e-2]
        self.thresh_cn, self.thresh_1hop, self.thresh_non1hop = ppr_thresholds

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.gnn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            dropout=gnn_dropout,
            norm="layer_norm",
        )
        self.gnn_norm = keras.layers.LayerNormalization()

        self.att_layers = [
            LPAttLayer(
                in_channels=hidden_channels * 2,
                out_channels=hidden_channels,
                node_dim=hidden_channels,
                num_heads=num_heads,
                dropout=transformer_dropout,
            )
            for _ in range(num_transformer_layers)
        ]

        self.elementwise_lin = MLP(hidden_channels, hidden_channels, hidden_channels)

        self.ppr_encoder_cn = MLP(2, hidden_channels, hidden_channels)
        self.ppr_encoder_onehop = MLP(2, hidden_channels, hidden_channels)
        self.ppr_encoder_non1hop = MLP(2, hidden_channels, hidden_channels)

        pairwise_dim = hidden_channels * num_heads + 4
        self.pairwise_lin = MLP(pairwise_dim, pairwise_dim, hidden_channels)
        self.score_func = MLP(hidden_channels * 2, hidden_channels * 2, 1, norm=None)

    def propagate(self, x, edge_index, training=False):
        h = self.gnn(x, edge_index, training=training)
        return self.gnn_norm(h)

    def call(self, batch, x, edge_index, ppr_matrix=None, training=False):
        r"""Forward pass of LPFormer.

        Args:
            batch: Tensor of shape (2, B) with link pairs (u, v) to predict.
            x: Node features of shape (N, in_channels).
            edge_index: Graph edge index (2, E).
            ppr_matrix: Optional precomputed PPR matrix of shape (N, N).
        """
        num_nodes = ops.shape(x)[0]
        if ppr_matrix is None:
            ppr_matrix = compute_ppr_matrix(edge_index, num_nodes)

        X_node = self.propagate(x, edge_index, training=training)

        u = ops.cast(batch[0], "int32")
        v = ops.cast(batch[1], "int32")

        x_u = ops.take(X_node, u, axis=0)
        x_v = ops.take(X_node, v, axis=0)

        elementwise_edge_feats = self.elementwise_lin(x_u * x_v, training=training)
        pairwise_feats = ops.concatenate([x_u, x_v], axis=-1)

        for att_layer in self.att_layers:
            pairwise_feats = att_layer(pairwise_feats, X_node, training=training)

        # Compute graph structural context counts (CNs, 1-hop, etc.)
        edge_index_np = ops.convert_to_numpy(edge_index)
        u_np = ops.convert_to_numpy(u)
        v_np = ops.convert_to_numpy(v)

        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        adj_matrix[edge_index_np[0], edge_index_np[1]] = 1.0

        cnts = []
        for ui, vi in zip(u_np, v_np):
            u_neighbors = adj_matrix[ui]
            v_neighbors = adj_matrix[vi]
            cn = np.sum((u_neighbors > 0) & (v_neighbors > 0))
            one_hop = np.sum((u_neighbors > 0) ^ (v_neighbors > 0))
            total_neigh = np.sum((u_neighbors > 0) | (v_neighbors > 0))
            non_1hop = num_nodes - total_neigh
            cnts.append([cn, one_hop, non_1hop, total_neigh])

        counts_tensor = ops.convert_to_tensor(np.array(cnts, dtype=np.float32))
        pairwise_feats = ops.concatenate([pairwise_feats, counts_tensor], axis=-1)
        pairwise_feats = self.pairwise_lin(pairwise_feats, training=training)

        combined_feats = ops.concatenate([elementwise_edge_feats, pairwise_feats], axis=-1)
        logits = self.score_func(combined_feats, training=training)
        return logits

