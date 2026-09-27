"""Multi-backend Keras 3 implementation of QET (Charge-Equilibration TensorNet)."""

from __future__ import annotations

import math
from typing import Sequence, Optional, Union, Tuple, Dict, Any, Literal
import keras
from keras import layers, ops
import numpy as np

from .core import MLP, scatter_add, infer_num_graphs
from .basis import compute_pair_vector_and_distance, polynomial_cutoff
from .tensornet import TensorNet
from .readout import ReduceReadOut


class LinearQeq(layers.Layer):
    """Closed-form charge-equilibration solver via Lagrange multipliers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._eps = 1e-12

    def call(self, chi, hardness, batch=None, total_charge=None, num_graphs=None):
        num_nodes = ops.shape(chi)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")
        n_graphs = infer_num_graphs(batch=batch, num_graphs=num_graphs)

        if total_charge is None:
            Q = ops.zeros((n_graphs,), dtype=chi.dtype)
        else:
            Q = ops.cast(total_charge, chi.dtype)
            if len(ops.shape(Q)) == 0:
                Q = ops.broadcast_to(Q, (n_graphs,))

        inv_eta = 1.0 / ops.maximum(hardness, 1e-6)
        chi_inv_eta = chi * inv_eta

        batch_clipped = ops.clip(batch, 0, ops.maximum(n_graphs - 1, 0))
        sum_inv_eta = ops.segment_sum(inv_eta, batch_clipped, num_segments=n_graphs)
        sum_chi_inv_eta = ops.segment_sum(chi_inv_eta, batch_clipped, num_segments=n_graphs)

        denom = ops.maximum(sum_inv_eta, self._eps)
        lambda_val = (Q + sum_chi_inv_eta) / denom
        lambda_per_node = ops.take(lambda_val, batch_clipped, axis=0)

        q_star = -chi_inv_eta + inv_eta * lambda_per_node
        return q_star


class ElectrostaticPotential(layers.Layer):
    """Gaussian-smeared Coulomb electrostatic potential calculation."""

    def __init__(self, cutoff: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self._inv_sqrt2 = float(1.0 / math.sqrt(2.0))

    def call(self, edge_index, charge, sigma, bond_dists, num_nodes: Optional[int] = None):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")

        num_c = ops.shape(charge)[0]
        src_c = ops.clip(src, 0, ops.maximum(num_c - 1, 0))
        dst_c = ops.clip(dst, 0, ops.maximum(num_c - 1, 0))

        q_j = ops.take(charge, dst_c, axis=0)
        s_i = ops.take(sigma, src_c, axis=0)
        s_j = ops.take(sigma, dst_c, axis=0)

        gamma_ij = ops.sqrt(ops.maximum(s_i ** 2 + s_j ** 2, 1e-8))
        d_safe = ops.maximum(bond_dists, 1e-7)
        arg = (d_safe * self._inv_sqrt2) / gamma_ij
        erf_val = ops.erf(arg)
        f_cut = polynomial_cutoff(bond_dists, self.cutoff, exponent=3)

        v_ij = (q_j / d_safe) * erf_val * f_cut
        return scatter_add(v_ij, src_c, num_segments=num_nodes)


class QET(TensorNet):
    """Charge-Equilibration TensorNet (QET) model."""

    def __init__(
        self,
        units: int = 64,
        nblocks: int = 2,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        ntypes_node: int = 95,
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__(
            units=units,
            nblocks=nblocks,
            num_rbf=num_rbf,
            cutoff=cutoff,
            ntypes_node=ntypes_node,
            activation_type=activation_type,
            **kwargs,
        )
        self.qeq_solver = LinearQeq()
        self.elec_pot = ElectrostaticPotential(cutoff=cutoff)

        self.chi_head = MLP([units, 32, 1], activation=activation_type, activate_last=False)
        self.hardness_head = MLP([units, 32, 1], activation=activation_type, activate_last=False)
        self.sigma_head = MLP([units, 32, 1], activation=activation_type, activate_last=False)

        self.energy_head = MLP([units + 2, units, 1], activation=activation_type, activate_last=False)
        self.qet_pool = ReduceReadOut(op="sum")

    def call(self, inputs, edge_index=None, node_type=None, pbc_offshift=None, batch=None, total_charge=None, num_graphs=None, state_attr=None):
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

        from .core import decompose_tensor, tensor_norm
        scalars, skew, traceless = decompose_tensor(X)
        norms = ops.concatenate([tensor_norm(scalars), tensor_norm(skew), tensor_norm(traceless)], axis=-1)
        node_feats = self.node_proj(self.out_norm(norms))

        chi = ops.squeeze(self.chi_head(node_feats), axis=-1)
        hardness = ops.softplus(ops.squeeze(self.hardness_head(node_feats), axis=-1)) + 0.1
        sigma = ops.softplus(ops.squeeze(self.sigma_head(node_feats), axis=-1)) + 0.1

        charges = self.qeq_solver(chi, hardness, batch=batch, total_charge=total_charge, num_graphs=n_graphs)
        v_elec = self.elec_pot(edge_index, charges, sigma, bond_dists, num_nodes=num_nodes)

        combined = ops.concatenate([node_feats, ops.expand_dims(charges, axis=-1), ops.expand_dims(v_elec, axis=-1)], axis=-1)
        e_atom = self.energy_head(combined)
        total_energy = self.qet_pool(e_atom, batch=batch, num_graphs=n_graphs)

        return ops.squeeze(total_energy, axis=-1)

