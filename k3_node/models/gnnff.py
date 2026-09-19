from typing import Optional
import numpy as np
import keras
from keras import ops

from k3_node.layers.aggr import SumAggregation
from k3_node.layers.pool import radius_graph
from k3_node.models.schnet import ShiftedSoftplus
from k3_node.models.dimenet import triplets


class GaussianFilter(keras.layers.Layer):
    r"""Gaussian filter for edge distances."""
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50, **kwargs):
        super().__init__(**kwargs)
        offset = np.linspace(start, stop, num_gaussians, dtype=np.float32)
        diff = float(offset[1] - offset[0])
        self.coeff = -0.5 / (diff ** 2)
        self.offset = self.add_weight(
            name="offset",
            shape=(num_gaussians,),
            initializer=keras.initializers.Constant(offset),
            trainable=False,
            dtype="float32",
        )

    def call(self, dist):
        dist = ops.expand_dims(dist, -1) - ops.expand_dims(self.offset, 0)
        return ops.exp(self.coeff * ops.power(dist, 2))


class NodeBlock(keras.layers.Layer):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_node_channels = hidden_node_channels
        self.lin_c1 = keras.layers.Dense(2 * hidden_node_channels)
        self.bn_c1 = keras.layers.BatchNormalization()
        self.bn = keras.layers.BatchNormalization()
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(self, node_emb, edge_emb, i):
        node_i = ops.take(node_emb, i, axis=0)
        c1 = ops.concatenate([node_i, edge_emb], axis=1)
        c1 = self.bn_c1(self.lin_c1(c1))
        c1_filter = ops.sigmoid(c1[:, :self.hidden_node_channels])
        c1_core = ops.tanh(c1[:, self.hidden_node_channels:])
        num_nodes = ops.shape(node_emb)[0]
        c1_emb = self.sum_aggr(c1_filter * c1_core, index=i, dim_size=num_nodes)
        c1_emb = self.bn(c1_emb)
        return ops.tanh(node_emb + c1_emb)


class EdgeBlock(keras.layers.Layer):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_edge_channels = hidden_edge_channels
        self.lin_c2 = keras.layers.Dense(2 * hidden_edge_channels)
        self.lin_c3 = keras.layers.Dense(2 * hidden_edge_channels)
        self.bn_c2 = keras.layers.BatchNormalization()
        self.bn_c3 = keras.layers.BatchNormalization()
        self.bn_c2_2 = keras.layers.BatchNormalization()
        self.bn_c3_2 = keras.layers.BatchNormalization()
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(
        self,
        node_emb,
        edge_emb,
        i,
        j,
        idx_i,
        idx_j,
        idx_k,
        idx_ji,
        idx_kj,
    ):
        node_i = ops.take(node_emb, i, axis=0)
        node_j = ops.take(node_emb, j, axis=0)
        c2 = node_i * node_j
        c2 = self.bn_c2(self.lin_c2(c2))
        c2_filter = ops.sigmoid(c2[:, :self.hidden_edge_channels])
        c2_core = ops.tanh(c2[:, self.hidden_edge_channels:])
        c2_emb = self.bn_c2_2(c2_filter * c2_core)

        node_idx_i = ops.take(node_emb, idx_i, axis=0)
        node_idx_j = ops.take(node_emb, idx_j, axis=0)
        node_idx_k = ops.take(node_emb, idx_k, axis=0)
        edge_idx_ji = ops.take(edge_emb, idx_ji, axis=0)
        edge_idx_kj = ops.take(edge_emb, idx_kj, axis=0)
        c3 = ops.concatenate([node_idx_i, node_idx_j, node_idx_k, edge_idx_ji, edge_idx_kj], axis=1)
        c3 = self.bn_c3(self.lin_c3(c3))
        c3_filter = ops.sigmoid(c3[:, :self.hidden_edge_channels])
        c3_core = ops.tanh(c3[:, self.hidden_edge_channels:])
        c3_emb = self.sum_aggr(c3_filter * c3_core, index=idx_ji, dim_size=ops.shape(edge_emb)[0])
        c3_emb = self.bn_c3_2(c3_emb)

        return ops.tanh(edge_emb + c2_emb + c3_emb)


class GNNFF(keras.layers.Layer):
    r"""The Graph Neural Network Force Field (GNNFF) from the
    `"Accurate and scalable graph neural network force field and molecular
    dynamics with direct force architecture"
    <https://www.nature.com/articles/s41524-021-00543-3>`_ paper.
    :class:`GNNFF` directly predicts atomic forces from automatically
    extracted features of the local atomic environment.

    Args:
        hidden_node_channels (int): Hidden node embedding size.
        hidden_edge_channels (int): Hidden edge embedding size.
        num_layers (int): Number of message passing blocks.
        cutoff (float, optional): Cutoff distance. (default: 5.0)
        max_num_neighbors (int, optional): Maximum neighbors per node. (default: 32)
    """
    def __init__(
        self,
        hidden_node_channels: int,
        hidden_edge_channels: int,
        num_layers: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.hidden_node_channels = hidden_node_channels
        self.hidden_edge_channels = hidden_edge_channels
        self.num_layers = num_layers

        self.node_emb = keras.Sequential([
            keras.layers.Embedding(95, hidden_node_channels),
            ShiftedSoftplus(),
            keras.layers.Dense(hidden_node_channels),
            ShiftedSoftplus(),
            keras.layers.Dense(hidden_node_channels),
        ])
        self.edge_emb = GaussianFilter(0.0, 5.0, hidden_edge_channels)

        self.node_blocks = [
            NodeBlock(hidden_node_channels, hidden_edge_channels)
            for _ in range(num_layers)
        ]
        self.edge_blocks = [
            EdgeBlock(hidden_node_channels, hidden_edge_channels)
            for _ in range(num_layers)
        ]

        self.force_predictor = keras.Sequential([
            keras.layers.Dense(hidden_edge_channels),
            ShiftedSoftplus(),
            keras.layers.Dense(hidden_edge_channels),
            ShiftedSoftplus(),
            keras.layers.Dense(1),
        ])
        self.sum_aggr = SumAggregation()

    def call(self, z, pos, batch=None):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )

        num_nodes = ops.shape(z)[0]
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=num_nodes)

        pos_i = ops.take(pos, i, axis=0)
        pos_j = ops.take(pos, j, axis=0)
        diff = pos_i - pos_j
        dist = ops.sqrt(ops.sum(ops.power(diff, 2), axis=-1))
        unit_vec = diff / ops.expand_dims(dist + 1e-8, -1)

        z = ops.cast(z, "int32")
        node_emb = self.node_emb(z)
        edge_emb = self.edge_emb(dist)

        for node_block, edge_block in zip(self.node_blocks, self.edge_blocks):
            node_emb = node_block(node_emb, edge_emb, i)
            edge_emb = edge_block(node_emb, edge_emb, i, j, idx_i, idx_j, idx_k, idx_ji, idx_kj)

        force = self.force_predictor(edge_emb) * unit_vec
        return self.sum_aggr(force, index=i, dim_size=num_nodes)

