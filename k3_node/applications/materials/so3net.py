"""Multi-backend Keras 3 implementation of SO3Net."""

from __future__ import annotations

import math
from typing import Sequence, Optional, Union, Tuple, Dict, Any, Literal
import keras
from keras import layers, ops
import numpy as np

from .core import MLP, scatter_add, infer_num_graphs
from .basis import (
    BondExpansion,
    RadialBesselFunction,
    compute_pair_vector_and_distance,
    cosine_cutoff,
)
from .readout import WeightedAtomReadOut, ReduceReadOut


class RealSphericalHarmonics(layers.Layer):
    """Computes real spherical harmonics up to order lmax for 3D unit vectors."""

    def __init__(self, lmax: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.lmax = lmax
        self.num_components = (lmax + 1) ** 2

    def call(self, vec):
        norm = ops.sqrt(ops.maximum(ops.sum(vec ** 2, axis=-1, keepdims=True), 1e-12))
        u = vec / norm
        x, y, z = u[..., 0:1], u[..., 1:2], u[..., 2:3]

        c00 = 0.5 * math.sqrt(1.0 / math.pi)
        y00 = ops.broadcast_to(ops.convert_to_tensor(c00, dtype=vec.dtype), ops.shape(x))
        comps = [y00]

        if self.lmax >= 1:
            c1 = math.sqrt(3.0 / (4.0 * math.pi))
            comps.extend([c1 * y, c1 * z, c1 * x])

        if self.lmax >= 2:
            c2_0 = 0.5 * math.sqrt(5.0 / (4.0 * math.pi))
            c2_1 = math.sqrt(15.0 / (4.0 * math.pi))
            c2_2 = 0.5 * math.sqrt(15.0 / (4.0 * math.pi))
            comps.extend([
                c2_1 * (x * y),
                c2_1 * (y * z),
                c2_0 * (3.0 * z ** 2 - 1.0),
                c2_1 * (x * z),
                c2_2 * (x ** 2 - y ** 2),
            ])

        return ops.concatenate(comps[:self.num_components], axis=-1)


class SO3Convolution(layers.Layer):
    """Equivariant interaction convolution layer for SO3Net."""

    def __init__(self, units: int, lmax: int = 2, num_rbf: int = 32, cutoff: float = 5.0, activation: str = "swish", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.lmax = lmax
        self.cutoff = float(cutoff)
        self.num_sh = (lmax + 1) ** 2
        self.rbf_proj = layers.Dense(units, use_bias=True)
        self.node_proj = layers.Dense(units, use_bias=True)
        self.sh_proj = layers.Dense(units, use_bias=False)
        self.out_proj = layers.Dense(units, use_bias=True)

    def call(self, edge_index, node_feats, edge_attr, sh_feats, edge_weight):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")
        num_nodes = ops.shape(node_feats)[0]

        C = ops.expand_dims(cosine_cutoff(edge_weight, self.cutoff), axis=-1)
        rbf_h = ops.silu(self.rbf_proj(edge_attr)) * C
        node_h = ops.take(self.node_proj(node_feats), dst, axis=0)

        scalar_msg = rbf_h * node_h
        msg = scatter_add(scalar_msg, src, num_segments=num_nodes)
        return node_feats + ops.silu(self.out_proj(msg))


class SO3Net(keras.Model):
    """SO(3)-equivariant representation model using spherical harmonics."""

    def __init__(
        self,
        units: int = 64,
        nblocks: int = 3,
        lmax: int = 2,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        ntypes_node: int = 95,
        ntargets: int = 1,
        readout_type: Literal["weighted_atom", "reduce_atom"] = "weighted_atom",
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.units = units
        self.lmax = lmax

        self.rbf = RadialBesselFunction(max_n=num_rbf, cutoff=cutoff)
        self.sh = RealSphericalHarmonics(lmax=lmax)
        self.embedding = layers.Embedding(ntypes_node, units)

        self.convs = [
            SO3Convolution(units=units, lmax=lmax, num_rbf=num_rbf, cutoff=cutoff, activation=activation_type)
            for _ in range(nblocks)
        ]

        if readout_type == "weighted_atom":
            self.readout = WeightedAtomReadOut(units, dims=[units, ntargets], activation=activation_type)
        else:
            self.readout = ReduceReadOut(op="mean")
            self.final_mlp = MLP([units, ntargets], activation=activation_type, activate_last=False)

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            pos = inputs.get("pos")
            edge_index = inputs.get("edge_index")
            node_type = inputs.get("node_type", inputs.get("z"))
            pbc_offshift = inputs.get("pbc_offshift", None)
            batch = inputs.get("batch", None)
            num_graphs = inputs.get("num_graphs", None)
            state_attr = inputs.get("state_attr", None)
            return pos, edge_index, node_type, pbc_offshift, batch, num_graphs, state_attr
        elif isinstance(inputs, (tuple, list)):
            pos = inputs[0]
            edge_index = inputs[1]
            node_type = inputs[2]
            pbc_offshift = inputs[3] if len(inputs) > 3 else None
            batch = inputs[4] if len(inputs) > 4 else None
            num_graphs = inputs[5] if len(inputs) > 5 else None
            state_attr = inputs[6] if len(inputs) > 6 else None
            return pos, edge_index, node_type, pbc_offshift, batch, num_graphs, state_attr
        return inputs, None, None, None, None, None, None

    def call(self, inputs, edge_index=None, node_type=None, pbc_offshift=None, batch=None, num_graphs=None, state_attr=None):
        if edge_index is None:
            (
                pos,
                edge_index,
                node_type_in,
                pbc_offshift_in,
                batch_in,
                num_graphs_in,
                state_attr_in,
            ) = self._unpack_inputs(inputs)
            if node_type is None:
                node_type = node_type_in
            if pbc_offshift is None:
                pbc_offshift = pbc_offshift_in
            if batch is None:
                batch = batch_in
            if num_graphs is None:
                num_graphs = num_graphs_in
            if state_attr is None:
                state_attr = state_attr_in
        else:
            pos = inputs

        num_nodes = ops.shape(pos)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")
        n_graphs = infer_num_graphs(batch=batch, num_graphs=num_graphs, state_attr=state_attr)

        vec, bond_dists = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        edge_attr = self.rbf(bond_dists)
        sh_attr = self.sh(vec)
        node_feats = self.embedding(ops.cast(node_type, "int32"))

        for conv in self.convs:
            node_feats = conv(edge_index, node_feats, edge_attr, sh_attr, bond_dists)

        if isinstance(self.readout, WeightedAtomReadOut):
            out = self.readout(node_feats, batch=batch, num_graphs=n_graphs)
        else:
            pooled = self.readout(node_feats, batch=batch, num_graphs=n_graphs)
            out = self.final_mlp(pooled)

        return ops.squeeze(out, axis=-1)
