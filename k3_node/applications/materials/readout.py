"""Readout layers for materials models."""

from __future__ import annotations

from typing import Sequence, Optional, Union
import keras
from keras import layers, ops
from k3_node.layers.pool import global_add_pool, global_mean_pool, global_max_pool
from .core import MLP, GatedMLP, get_activation, infer_num_graphs


class ReduceReadOut(layers.Layer):
    """Pool node features into graph features via sum, mean, or max reduction."""

    def __init__(self, op: str = "mean", field: str = "node_feat", **kwargs):
        super().__init__(**kwargs)
        self.op = op.lower()
        self.field = field

    def call(self, node_feat, batch=None, num_graphs=None):
        n_graphs = infer_num_graphs(batch=batch, num_graphs=num_graphs)
        if self.op == "mean":
            return global_mean_pool(node_feat, batch=batch, size=n_graphs)
        elif self.op == "sum":
            return global_add_pool(node_feat, batch=batch, size=n_graphs)
        elif self.op == "max":
            return global_max_pool(node_feat, batch=batch, size=n_graphs)
        else:
            raise ValueError(f"Unsupported reduction op: {self.op}")


class WeightedReadOut(layers.Layer):
    """Feed node features through a GatedMLP to predict atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.in_feats = in_feats
        self.dims = list(dims)
        self.num_targets = num_targets
        self.gated = GatedMLP(in_feats=in_feats, dims=[*self.dims, num_targets], activate_last=False)

    def call(self, node_feat):
        return self.gated(node_feat)


class WeightedAtomReadOut(layers.Layer):
    """Weighted atom readout for whole-graph properties with normalized learned weights."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activation: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feats = in_feats
        self.dims = list(dims)
        self.mlp = MLP([in_feats, *self.dims], activation=activation, activate_last=True)
        self.weight_mlp = MLP([in_feats, *self.dims[:-1], 1], activation=activation, activate_last=False)

    def call(self, node_feat, batch=None, num_graphs=None):
        if batch is None:
            batch = ops.zeros((ops.shape(node_feat)[0],), dtype="int32")
        batch = ops.cast(batch, "int32")
        n_graphs = infer_num_graphs(batch=batch, num_graphs=num_graphs)

        updated_field = self.mlp(node_feat)
        weights = ops.sigmoid(self.weight_mlp(node_feat))  # [num_nodes, 1]

        weight_sum = ops.segment_sum(weights, batch, num_segments=n_graphs)  # [num_graphs, 1]
        ws_len = ops.shape(weight_sum)[0]
        b_safe = ops.clip(batch, 0, ops.maximum(ws_len - 1, 0))
        weight_sum_per_node = ops.take(weight_sum, b_safe, axis=0)
        factor = weights / ops.maximum(weight_sum_per_node, 1e-8)  # [num_nodes, 1]

        return global_add_pool(factor * updated_field, batch=batch, size=n_graphs)


class Set2SetReadOut(layers.Layer):
    """Iterative content-based attention pooling (Set2Set) for nodes."""

    def __init__(self, in_channels: int, processing_steps: int = 3, num_layers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm_cell = layers.LSTMCell(in_channels)

    def build(self, input_shape=None):
        self.lstm_cell.build((None, self.out_channels))
        super().build(input_shape)

    def call(self, x, batch=None, num_graphs=None):
        if batch is None:
            batch = ops.zeros((ops.shape(x)[0],), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")
        n_graphs = infer_num_graphs(batch=batch, num_graphs=num_graphs)

        h = [
            ops.zeros((n_graphs, self.in_channels), dtype=x.dtype),
            ops.zeros((n_graphs, self.in_channels), dtype=x.dtype),
        ]
        q_star = ops.zeros((n_graphs, self.out_channels), dtype=x.dtype)
        b_safe = ops.clip(batch, 0, ops.maximum(n_graphs - 1, 0))

        for _ in range(self.processing_steps):
            q, h = self.lstm_cell(q_star, h)
            q_taken = ops.take(q, b_safe, axis=0)
            e = ops.sum(x * q_taken, axis=-1, keepdims=True)

            max_e = ops.segment_max(e, batch, num_segments=n_graphs)
            max_e_taken = ops.take(max_e, b_safe, axis=0)
            exp_e = ops.exp(e - max_e_taken)

            sum_exp_e = ops.segment_sum(exp_e, batch, num_segments=n_graphs)
            sum_exp_e_taken = ops.take(sum_exp_e, b_safe, axis=0)
            alpha = exp_e / ops.maximum(sum_exp_e_taken, 1e-12)

            r = ops.segment_sum(alpha * x, batch, num_segments=n_graphs)
            q_star = ops.concatenate([q, r], axis=-1)

        return q_star


class EdgeSet2Set(layers.Layer):
    """Iterative content-based attention pooling (Set2Set) for edge features."""

    def __init__(self, input_dim: int, n_iters: int = 3, n_layers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm_cell = layers.LSTMCell(input_dim)

    def build(self, input_shape=None):
        self.lstm_cell.build((None, self.output_dim))
        super().build(input_shape)

    def call(self, edge_feat, edge_batch=None, num_graphs=None):
        if edge_batch is None:
            edge_batch = ops.zeros((ops.shape(edge_feat)[0],), dtype="int32")
        else:
            edge_batch = ops.cast(edge_batch, "int32")
        n_graphs = infer_num_graphs(batch=edge_batch, num_graphs=num_graphs)

        h = [
            ops.zeros((n_graphs, self.input_dim), dtype=edge_feat.dtype),
            ops.zeros((n_graphs, self.input_dim), dtype=edge_feat.dtype),
        ]
        q_star = ops.zeros((n_graphs, self.output_dim), dtype=edge_feat.dtype)
        eb_safe = ops.clip(edge_batch, 0, ops.maximum(n_graphs - 1, 0))

        for _ in range(self.n_iters):
            q, h = self.lstm_cell(q_star, h)
            q_taken = ops.take(q, eb_safe, axis=0)
            e = ops.sum(edge_feat * q_taken, axis=-1, keepdims=True)

            max_e = ops.segment_max(e, edge_batch, num_segments=n_graphs)
            max_e_taken = ops.take(max_e, eb_safe, axis=0)
            exp_e = ops.exp(e - max_e_taken)

            sum_exp_e = ops.segment_sum(exp_e, edge_batch, num_segments=n_graphs)
            sum_exp_e_taken = ops.take(sum_exp_e, eb_safe, axis=0)
            alpha = exp_e / ops.maximum(sum_exp_e_taken, 1e-12)

            r = ops.segment_sum(alpha * edge_feat, edge_batch, num_segments=n_graphs)
            q_star = ops.concatenate([q, r], axis=-1)

        return q_star

