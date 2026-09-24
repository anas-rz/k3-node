"""Multi-backend Keras 3 implementation of M3GNet."""

from __future__ import annotations

from typing import Sequence, Optional, Union, Tuple, Dict, Any, Literal
import keras
from keras import layers, ops
import numpy as np

from .core import MLP, GatedMLP, EmbeddingBlock, scatter_add, scatter_mean, infer_num_graphs
from .basis import (
    BondExpansion,
    SphericalBesselWithHarmonics,
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    cosine_cutoff,
)
from .readout import WeightedAtomReadOut, ReduceReadOut, Set2SetReadOut


class ThreeBodyInteractions(layers.Layer):
    """Three-body bond angular update using directed line-graph message passing."""

    def __init__(
        self,
        update_network_atom: layers.Layer,
        update_network_bond: layers.Layer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def call(
        self,
        edge_index,
        line_edge_index,
        three_basis,
        three_cutoff,
        node_feat,
        edge_feat,
    ):
        src_bond = ops.cast(line_edge_index[0], "int32")
        dst_bond = ops.cast(line_edge_index[1], "int32")
        edge_dst_atom = ops.cast(edge_index[1], "int32")
        num_bonds = ops.shape(edge_feat)[0]

        # Destination atom of destination bond
        end_atom_idx = ops.take(edge_dst_atom, dst_bond, axis=0)
        updated_atoms = self.update_network_atom(node_feat)
        end_atom_features = ops.take(updated_atoms, end_atom_idx, axis=0)

        basis = three_basis * end_atom_features
        weights = ops.take(three_cutoff, src_bond, axis=0) * ops.take(three_cutoff, dst_bond, axis=0)
        basis = basis * ops.expand_dims(weights, axis=-1)

        new_bonds = scatter_add(basis, src_bond, num_segments=num_bonds)
        return edge_feat + self.update_network_bond(new_bonds)


class M3GNetGraphConv(layers.Layer):
    """M3GNet graph convolution layer: two-body edge and node updates."""

    def __init__(
        self,
        degree: int,
        edge_dims: Sequence[int],
        node_dims: Sequence[int],
        state_dims: Optional[Sequence[int]] = None,
        include_state: bool = False,
        activation: str = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_state = include_state
        self.edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        self.edge_weight_func = layers.Dense(edge_dims[-1], use_bias=False)

        self.node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        self.node_weight_func = layers.Dense(node_dims[-1], use_bias=False)

        if include_state and state_dims is not None:
            self.state_update_func = MLP(state_dims, activation=activation, activate_last=True)
        else:
            self.state_update_func = None

    def call(
        self,
        edge_index,
        edge_feat,
        node_feat,
        state_feat,
        rbf,
        batch=None,
        edge_batch=None,
        num_nodes: Optional[int] = None,
        num_graphs: Optional[int] = None,
    ):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")

        vi = ops.take(node_feat, src, axis=0)
        vj = ops.take(node_feat, dst, axis=0)

        if self.include_state and state_feat is not None:
            if edge_batch is not None:
                u_edge = ops.take(state_feat, edge_batch, axis=0)
            else:
                num_e = ops.shape(edge_feat)[0]
                u_edge = ops.broadcast_to(state_feat, (num_e, ops.shape(state_feat)[-1]))
            edge_inputs = ops.concatenate([vi, vj, edge_feat, u_edge], axis=-1)
        else:
            edge_inputs = ops.concatenate([vi, vj, edge_feat], axis=-1)

        # 1. Edge update
        edge_update = self.edge_update_func(edge_inputs) * self.edge_weight_func(rbf)
        edge_feat_new = edge_feat + edge_update

        # 2. Node update
        node_update = self.node_update_func(edge_inputs) * self.node_weight_func(rbf)
        node_update_sum = scatter_add(node_update, src, num_segments=num_nodes)
        node_feat_new = node_feat + node_update_sum

        # 3. State update
        state_feat_new = state_feat
        if self.include_state and self.state_update_func is not None and state_feat is not None:
            if batch is not None:
                uv = scatter_mean(node_feat_new, batch, num_segments=num_graphs)
            else:
                uv = ops.mean(node_feat_new, axis=0, keepdims=True)
            state_inputs = ops.concatenate([state_feat, uv], axis=-1)
            state_feat_new = self.state_update_func(state_inputs)

        return edge_feat_new, node_feat_new, state_feat_new


class M3GNetBlock(layers.Layer):
    """M3GNet block wrapping M3GNetGraphConv with optional dropout."""

    def __init__(
        self,
        degree: int,
        conv_hiddens: Sequence[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        activation: str = "swish",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_state = include_state
        edge_in = 2 * dim_node_feats + dim_edge_feats + (dim_state_feats if include_state else 0)
        node_in = 2 * dim_node_feats + dim_edge_feats + (dim_state_feats if include_state else 0)
        state_in = dim_state_feats + dim_node_feats if include_state else 0

        self.conv = M3GNetGraphConv(
            degree=degree,
            edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
            node_dims=[node_in, *conv_hiddens, dim_node_feats],
            state_dims=[state_in, *conv_hiddens, dim_state_feats] if include_state else None,
            include_state=include_state,
            activation=activation,
        )
        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def call(
        self,
        edge_index,
        edge_feat,
        node_feat,
        state_feat,
        rbf,
        batch=None,
        edge_batch=None,
        num_nodes: Optional[int] = None,
        num_graphs: Optional[int] = None,
    ):
        edge_feat, node_feat, state_feat = self.conv(
            edge_index, edge_feat, node_feat, state_feat, rbf,
            batch=batch, edge_batch=edge_batch, num_nodes=num_nodes, num_graphs=num_graphs
        )
        if self.dropout is not None:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
        return edge_feat, node_feat, state_feat


class M3GNet(keras.Model):
    """M3GNet materials potential model supporting 3-body angles and multibackend training."""

    def __init__(
        self,
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: Optional[int] = None,
        max_n: int = 3,
        max_l: int = 3,
        nblocks: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",
        is_intensive: bool = True,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        include_state: bool = False,
        activation_type: str = "swish",
        dropout: float = 0.0,
        ntypes_node: int = 95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.is_intensive = is_intensive

        self.bond_expansion = BondExpansion(
            max_l=max_l,
            max_n=max_n,
            cutoff=cutoff,
            rbf_type=rbf_type,
            smooth=False,
        )

        degree_rbf = max_n * max_l if rbf_type.lower() == "sphericalbessel" else 100
        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=ntypes_node,
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding if include_state else None,
            activation=activation_type,
        )

        self.sbf_shf = SphericalBesselWithHarmonics(
            max_n=max_n,
            max_l=max_l,
            cutoff=threebody_cutoff,
        )
        degree_3body = max_n * max_l

        self.three_body_interactions = []
        self.blocks = []
        for _ in range(nblocks):
            self.three_body_interactions.append(
                ThreeBodyInteractions(
                    update_network_atom=MLP([dim_node_embedding, degree_3body], activation=activation_type, activate_last=False),
                    update_network_bond=GatedMLP(in_feats=degree_3body, dims=[dim_edge_embedding], activate_last=False),
                )
            )
            self.blocks.append(
                M3GNetBlock(
                    degree=degree_rbf,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_embedding if include_state else 0,
                    include_state=include_state,
                    activation=activation_type,
                    dropout=dropout,
                )
            )

        if readout_type == "weighted_atom":
            self.readout = WeightedAtomReadOut(dim_node_embedding, dims=[units, units, ntargets], activation=activation_type)
        elif readout_type == "set2set":
            self.readout = Set2SetReadOut(dim_node_embedding)
            self.final_mlp = MLP([2 * dim_node_embedding, units, ntargets], activation=activation_type, activate_last=False)
        else:
            self.readout = ReduceReadOut(op="mean")
            self.final_mlp = MLP([dim_node_embedding, units, ntargets], activation=activation_type, activate_last=False)

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            pos = inputs.get("pos")
            edge_index = inputs.get("edge_index")
            node_type = inputs.get("node_type", inputs.get("z"))
            line_edge_index = inputs.get("line_edge_index", None)
            state_attr = inputs.get("state_attr", None)
            pbc_offshift = inputs.get("pbc_offshift", None)
            batch = inputs.get("batch", None)
            num_graphs = inputs.get("num_graphs", None)
            return pos, edge_index, node_type, line_edge_index, state_attr, pbc_offshift, batch, num_graphs
        elif isinstance(inputs, (tuple, list)):
            pos = inputs[0]
            edge_index = inputs[1]
            node_type = inputs[2]
            line_edge_index = inputs[3] if len(inputs) > 3 else None
            state_attr = inputs[4] if len(inputs) > 4 else None
            pbc_offshift = inputs[5] if len(inputs) > 5 else None
            batch = inputs[6] if len(inputs) > 6 else None
            num_graphs = inputs[7] if len(inputs) > 7 else None
            return pos, edge_index, node_type, line_edge_index, state_attr, pbc_offshift, batch, num_graphs
        return inputs, None, None, None, None, None, None, None

    def call(
        self,
        inputs,
        edge_index=None,
        node_type=None,
        line_edge_index=None,
        state_attr=None,
        pbc_offshift=None,
        batch=None,
        num_graphs=None,
    ):
        if edge_index is None:
            (
                pos,
                edge_index,
                node_type_in,
                line_edge_index_in,
                state_attr_in,
                pbc_offshift_in,
                batch_in,
                num_graphs_in,
            ) = self._unpack_inputs(inputs)
            if node_type is None:
                node_type = node_type_in
            if line_edge_index is None:
                line_edge_index = line_edge_index_in
            if state_attr is None:
                state_attr = state_attr_in
            if pbc_offshift is None:
                pbc_offshift = pbc_offshift_in
            if batch is None:
                batch = batch_in
            if num_graphs is None:
                num_graphs = num_graphs_in
        else:
            pos = inputs

        num_nodes = ops.shape(pos)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")
        n_graphs = infer_num_graphs(batch=batch, num_graphs=num_graphs, state_attr=state_attr)

        src = ops.cast(edge_index[0], "int32")
        edge_batch = ops.take(batch, src, axis=0)

        # 1. 2-body pair distances and expansion
        _, bond_dists = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        edge_attr = self.bond_expansion(bond_dists)

        # 2. Embeddings
        node_feat, edge_feat, state_feat = self.embedding(node_type, edge_attr, state_attr)

        # 3. 3-body expansion if line_edge_index is available
        three_cutoff = cosine_cutoff(bond_dists, self.threebody_cutoff)
        if line_edge_index is not None and ops.shape(line_edge_index)[1] > 0:
            theta, phi = compute_theta_and_phi(pos, edge_index, line_edge_index, pbc_offshift)
            src_bonds = ops.cast(line_edge_index[0], "int32")
            r_triplets = ops.take(bond_dists, src_bonds, axis=0)
            three_basis = self.sbf_shf(r_triplets, theta, phi)
        else:
            three_basis = None

        # 4. Message passing blocks
        for i in range(len(self.blocks)):
            if three_basis is not None and line_edge_index is not None:
                edge_feat = self.three_body_interactions[i](
                    edge_index, line_edge_index, three_basis, three_cutoff, node_feat, edge_feat
                )
            edge_feat, node_feat, state_feat = self.blocks[i](
                edge_index, edge_feat, node_feat, state_feat, edge_attr,
                batch=batch, edge_batch=edge_batch, num_nodes=num_nodes, num_graphs=n_graphs
            )

        # 5. Readout
        if isinstance(self.readout, WeightedAtomReadOut):
            output = self.readout(node_feat, batch=batch, num_graphs=n_graphs)
        else:
            pooled = self.readout(node_feat, batch=batch, num_graphs=n_graphs)
            output = self.final_mlp(pooled)

        return ops.squeeze(output, axis=-1)

