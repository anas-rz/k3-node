"""Multi-backend Keras 3 implementation of TensorNet."""

from __future__ import annotations

from typing import Sequence, Optional, Union, Tuple, Dict, Any, Literal
import keras
from keras import layers, ops
import numpy as np

from .core import (
    MLP,
    vector_to_skewtensor,
    vector_to_symtensor,
    decompose_tensor,
    tensor_norm,
    scatter_add,
    infer_num_graphs,
)
from .basis import (
    BondExpansion,
    RadialBesselFunction,
    compute_pair_vector_and_distance,
    cosine_cutoff,
)
from .readout import WeightedAtomReadOut, ReduceReadOut


class TensorEmbedding(layers.Layer):
    """Embeds node types and Cartesian pair vectors into rank-2 tensors [num_nodes, units, 3, 3]."""

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        ntypes_node: int = 95,
        cutoff: float = 5.0,
        activation: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.cutoff = cutoff

        self.distance_proj1 = layers.Dense(units, use_bias=True)
        self.distance_proj2 = layers.Dense(units, use_bias=True)
        self.distance_proj3 = layers.Dense(units, use_bias=True)

        self.emb = layers.Embedding(ntypes_node, units)
        self.emb2 = layers.Dense(units, use_bias=True)

        self.linears_tensor = [layers.Dense(units, use_bias=False) for _ in range(3)]
        self.linears_scalar = [
            layers.Dense(2 * units, use_bias=True),
            layers.Dense(3 * units, use_bias=True),
        ]
        self.init_norm = layers.LayerNormalization(axis=-1)

    def call(self, z, edge_index, edge_attr, edge_weight, vec):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")
        num_nodes = ops.shape(z)[0]

        # Normalized pair vectors
        vec_norm = vec / ops.maximum(ops.expand_dims(edge_weight, axis=-1), 1e-7)

        # Distance projections: [num_edges, units]
        C = ops.expand_dims(cosine_cutoff(edge_weight, self.cutoff), axis=-1)
        f_I = self.distance_proj1(edge_attr) * C
        f_A = self.distance_proj2(edge_attr) * C
        f_S = self.distance_proj3(edge_attr) * C

        # Geometric tensors: [num_edges, 3, 3]
        I_mat = ops.expand_dims(ops.eye(3, dtype=vec.dtype), axis=0)  # [1, 3, 3]
        A_mat = vector_to_skewtensor(vec_norm)  # [num_edges, 3, 3]
        S_mat = vector_to_symtensor(vec_norm)  # [num_edges, 3, 3]

        # Expand to units dimension: [num_edges, units, 3, 3]
        Iij = ops.expand_dims(ops.expand_dims(f_I, axis=-1), axis=-1) * ops.expand_dims(I_mat, axis=1)
        Aij = ops.expand_dims(ops.expand_dims(f_A, axis=-1), axis=-1) * ops.expand_dims(A_mat, axis=1)
        Sij = ops.expand_dims(ops.expand_dims(f_S, axis=-1), axis=-1) * ops.expand_dims(S_mat, axis=1)

        # Node chemical embeddings
        node_emb = self.emb(ops.cast(z, "int32"))
        vi = ops.take(node_emb, src, axis=0)
        vj = ops.take(node_emb, dst, axis=0)
        zij = ops.concatenate([vi, vj], axis=-1)
        Zij = ops.expand_dims(ops.expand_dims(self.emb2(zij), axis=-1), axis=-1)

        scalars_msg = Zij * Iij
        skew_msg = Zij * Aij
        traceless_msg = Zij * Sij

        scalars = scatter_add(scalars_msg, src, num_segments=num_nodes)
        skew = scatter_add(skew_msg, src, num_segments=num_nodes)
        traceless = scatter_add(traceless_msg, src, num_segments=num_nodes)

        # Apply tensor linear transformations: transpose to apply Dense along units axis
        s_t = ops.transpose(scalars, [0, 2, 3, 1])
        a_t = ops.transpose(skew, [0, 2, 3, 1])
        tr_t = ops.transpose(traceless, [0, 2, 3, 1])

        scalars = ops.transpose(self.linears_tensor[0](s_t), [0, 3, 1, 2])
        skew = ops.transpose(self.linears_tensor[1](a_t), [0, 3, 1, 2])
        traceless = ops.transpose(self.linears_tensor[2](tr_t), [0, 3, 1, 2])

        # Node invariant scalar feature
        s_norm = self.init_norm(tensor_norm(scalars))
        s_norm = ops.silu(self.linears_scalar[0](s_norm))
        s_norm = self.linears_scalar[1](s_norm)

        f_I_node = s_norm[..., :self.units]
        f_A_node = s_norm[..., self.units:2 * self.units]
        f_S_node = s_norm[..., 2 * self.units:]

        scalars = ops.expand_dims(ops.expand_dims(f_I_node, axis=-1), axis=-1) * scalars
        skew = ops.expand_dims(ops.expand_dims(f_A_node, axis=-1), axis=-1) * skew
        traceless = ops.expand_dims(ops.expand_dims(f_S_node, axis=-1), axis=-1) * traceless

        X = scalars + skew + traceless
        return X


class TensorNetInteraction(layers.Layer):
    """Equivariant Cartesian tensor message passing interaction layer."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        cutoff: float = 5.0,
        activation: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff

        self.linears_scalar = [
            layers.Dense(units, use_bias=True),
            layers.Dense(2 * units, use_bias=True),
            layers.Dense(3 * units, use_bias=True),
        ]
        self.linears_tensor = [layers.Dense(units, use_bias=False) for _ in range(6)]

    def call(self, edge_index, edge_weight, edge_attr, X):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")
        num_nodes = ops.shape(X)[0]

        # Process edge attributes
        C = ops.expand_dims(cosine_cutoff(edge_weight, self.cutoff), axis=-1)
        h_edge = edge_attr
        for linear in self.linears_scalar:
            h_edge = ops.silu(linear(h_edge))
        edge_attr_processed = h_edge * C

        f_I = edge_attr_processed[..., :self.units]
        f_A = edge_attr_processed[..., :self.units]
        f_S = edge_attr_processed[..., :self.units]

        # Normalize input tensor
        X_norm = ops.expand_dims(ops.expand_dims(tensor_norm(X) + 1.0, axis=-1), axis=-1)
        X_normalized = X / X_norm

        scalars, skew, traceless = decompose_tensor(X_normalized)

        # Tensor linears (0, 1, 2)
        s_t = ops.transpose(scalars, [0, 2, 3, 1])
        a_t = ops.transpose(skew, [0, 2, 3, 1])
        tr_t = ops.transpose(traceless, [0, 2, 3, 1])

        scalars = ops.transpose(self.linears_tensor[0](s_t), [0, 3, 1, 2])
        skew = ops.transpose(self.linears_tensor[1](a_t), [0, 3, 1, 2])
        traceless = ops.transpose(self.linears_tensor[2](tr_t), [0, 3, 1, 2])

        # Gather node features for edges
        sc_j = ops.take(scalars, dst, axis=0)
        sk_j = ops.take(skew, dst, axis=0)
        tr_j = ops.take(traceless, dst, axis=0)

        # Modulate by radial features
        msg_s = ops.expand_dims(ops.expand_dims(f_I, axis=-1), axis=-1) * sc_j
        msg_a = ops.expand_dims(ops.expand_dims(f_A, axis=-1), axis=-1) * sk_j
        msg_tr = ops.expand_dims(ops.expand_dims(f_S, axis=-1), axis=-1) * tr_j

        msg = msg_s + msg_a + msg_tr
        X_update = scatter_add(msg, src, num_segments=num_nodes)

        # Tensor linears (3, 4, 5)
        s2, a2, tr2 = decompose_tensor(X_update)
        s2_t = ops.transpose(s2, [0, 2, 3, 1])
        a2_t = ops.transpose(a2, [0, 2, 3, 1])
        tr2_t = ops.transpose(tr2, [0, 2, 3, 1])

        s2 = ops.transpose(self.linears_tensor[3](s2_t), [0, 3, 1, 2])
        a2 = ops.transpose(self.linears_tensor[4](a2_t), [0, 3, 1, 2])
        tr2 = ops.transpose(self.linears_tensor[5](tr2_t), [0, 3, 1, 2])

        return X + s2 + a2 + tr2


class TensorNet(keras.Model):
    """Cartesian tensor-based equivariant GNN for molecular and crystal potentials."""

    def __init__(
        self,
        units: int = 64,
        nblocks: int = 2,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        ntypes_node: int = 95,
        ntargets: int = 1,
        readout_type: Literal["weighted_atom", "reduce_atom"] = "weighted_atom",
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.cutoff = cutoff

        if rbf_type.lower() == "gaussian":
            self.bond_expansion = BondExpansion(
                rbf_type="Gaussian",
                initial=0.0,
                final=cutoff,
                num_centers=num_rbf,
            )
        else:
            self.bond_expansion = RadialBesselFunction(max_n=num_rbf, cutoff=cutoff)

        self.embedding = TensorEmbedding(
            units=units,
            degree_rbf=num_rbf,
            ntypes_node=ntypes_node,
            cutoff=cutoff,
            activation=activation_type,
        )

        self.interactions = [
            TensorNetInteraction(num_rbf=num_rbf, units=units, cutoff=cutoff, activation=activation_type)
            for _ in range(nblocks)
        ]

        self.out_norm = layers.LayerNormalization(axis=-1)
        self.node_proj = MLP([3 * units, units, units], activation=activation_type, activate_last=True)

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
        edge_attr = self.bond_expansion(bond_dists)

        X = self.embedding(node_type, edge_index, edge_attr, bond_dists, vec)

        for interaction in self.interactions:
            X = interaction(edge_index, bond_dists, edge_attr, X)

        # Decompose into irreducible norms
        scalars, skew, traceless = decompose_tensor(X)
        norm_s = tensor_norm(scalars)
        norm_a = tensor_norm(skew)
        norm_tr = tensor_norm(traceless)

        norms = ops.concatenate([norm_s, norm_a, norm_tr], axis=-1)
        node_feats = self.node_proj(self.out_norm(norms))

        if isinstance(self.readout, WeightedAtomReadOut):
            out = self.readout(node_feats, batch=batch, num_graphs=n_graphs)
        else:
            pooled = self.readout(node_feats, batch=batch, num_graphs=n_graphs)
            out = self.final_mlp(pooled)

        return ops.squeeze(out, axis=-1)

