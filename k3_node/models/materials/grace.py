"""Multi-backend Keras 3 implementation of GRACE (Graph Atomic Cluster Expansion)."""

from __future__ import annotations

from typing import Sequence, Optional, Union, Tuple, Dict, Any, Literal
import keras
from keras import layers, ops
import numpy as np

from .core import MLP, scatter_add, infer_num_graphs
from .basis import (
    ChebyshevRadialBasis,
    compute_pair_vector_and_distance,
    polynomial_cutoff,
)
from .so3net import RealSphericalHarmonics
from .readout import ReduceReadOut


class GraceSPBasis(layers.Layer):
    """Single-particle ACE basis aggregation for atomic clusters."""

    def __init__(
        self,
        n_rad_base: int = 6,
        lmax: int = 2,
        embedding_size: int = 8,
        ntypes_node: int = 95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rad_base = n_rad_base
        self.lmax = lmax
        self.num_sh = (lmax + 1) ** 2
        self.elem_emb = layers.Embedding(ntypes_node, embedding_size)
        self.radial_proj = layers.Dense(embedding_size, use_bias=False)

    def call(self, edge_index, node_type, rad_basis, sh_basis, num_nodes: Optional[int] = None):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")

        # Neighbor chemical indicator
        neigh_z = ops.take(node_type, dst, axis=0)
        c_j = self.elem_emb(ops.cast(neigh_z, "int32"))

        # Radial modulation
        R_nl = self.radial_proj(rad_basis)
        radial_atom = c_j * R_nl

        # Outer product with spherical harmonics: [E, num_sh, emb_size]
        sp_msg = ops.expand_dims(sh_basis, axis=-1) * ops.expand_dims(radial_atom, axis=1)

        # Aggregate into central atom
        A_i = scatter_add(sp_msg, src, num_segments=num_nodes)
        return A_i


class GraceACEStack(layers.Layer):
    """Multi-order Atomic Cluster Expansion stack accumulating rotational invariants."""

    def __init__(
        self,
        embedding_size: int = 8,
        max_order: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.max_order = max_order

    def call(self, A_i):
        A_00 = A_i[:, 0, :]
        invariants = [A_00]

        if self.max_order >= 2:
            A2 = ops.sum(A_i ** 2, axis=1)
            invariants.append(A2)

        if self.max_order >= 3:
            A3 = A_00 * ops.sum(A_i ** 2, axis=1)
            invariants.append(A3)

        return ops.concatenate(invariants, axis=-1)


class GRACE(keras.Model):
    """Graph Atomic Cluster Expansion (GRACE) foundational interatomic potential."""

    def __init__(
        self,
        cutoff: float = 5.0,
        n_rad_base: int = 6,
        lmax: int = 2,
        embedding_size: int = 8,
        max_order: int = 3,
        nblocks: int = 2,
        readout_hidden: Sequence[int] = (32,),
        ntypes_node: int = 95,
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.nblocks = nblocks

        self.chebyshev = ChebyshevRadialBasis(nfunc=n_rad_base, cutoff=cutoff, cutoff_exponent=5)
        self.sh = RealSphericalHarmonics(lmax=lmax)

        self.sp_bases = []
        self.ace_stacks = []
        self.readouts = []
        inv_dim = embedding_size * min(max_order, 3)

        for _ in range(nblocks):
            self.sp_bases.append(
                GraceSPBasis(
                    n_rad_base=n_rad_base,
                    lmax=lmax,
                    embedding_size=embedding_size,
                    ntypes_node=ntypes_node,
                )
            )
            self.ace_stacks.append(
                GraceACEStack(
                    embedding_size=embedding_size,
                    max_order=max_order,
                )
            )
            self.readouts.append(
                MLP([inv_dim, *readout_hidden, 1], activation=activation_type, activate_last=False)
            )

        self.graph_pool = ReduceReadOut(op="sum")

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
        rad_basis = self.chebyshev(bond_dists)
        sh_basis = self.sh(vec)

        total_atomic_energy = ops.zeros((num_nodes, 1), dtype=pos.dtype)
        for i in range(self.nblocks):
            A_i = self.sp_bases[i](edge_index, node_type, rad_basis, sh_basis, num_nodes=num_nodes)
            invariants = self.ace_stacks[i](A_i)
            e_block = self.readouts[i](invariants)
            total_atomic_energy = total_atomic_energy + e_block

        total_energy = self.graph_pool(total_atomic_energy, batch=batch, num_graphs=n_graphs)
        return ops.squeeze(total_energy, axis=-1)

