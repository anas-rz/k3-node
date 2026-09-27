"""Multi-backend Keras 3 implementation of CHGNet."""

from __future__ import annotations

import math
from typing import Sequence, Optional, Union, Tuple, Dict, Any, Literal
import keras
from keras import layers, ops
import numpy as np

from .core import MLP, GatedMLP, scatter_add, scatter_mean, infer_num_graphs
from .basis import (
    RadialBesselFunction,
    FourierExpansion,
    compute_pair_vector_and_distance,
    compute_theta,
    polynomial_cutoff,
)
from .readout import ReduceReadOut
from k3_node.hub.hub_mixin import K3NodeHubMixin


class CHGNetAtomGraphBlock(layers.Layer):
    """Atom-graph convolution block for CHGNet.

    Example:
        ```python
        import numpy as np
        from k3_node.models import CHGNetAtomGraphBlock

        edge_index = np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])  # 4 atoms, 8 bonds
        node_feat = np.random.rand(4, 16).astype("float32")
        edge_feat = np.random.rand(8, 16).astype("float32")

        block = CHGNetAtomGraphBlock(num_atom_feats=16, num_bond_feats=16, atom_hidden_dims=(16,))
        atom_out, bond_out = block(edge_index, node_feat, edge_feat)  # updated atom and bond features
        print(tuple(atom_out.shape), tuple(bond_out.shape))  # (4, 16) (8, 16)
        ```
    """

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        atom_hidden_dims: Sequence[int] = (64,),
        bond_hidden_dims: Optional[Sequence[int]] = (64,),
        activation: str = "swish",
        normalization: Optional[str] = "layer",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        node_input_dim = 2 * num_atom_feats + num_bond_feats
        self.node_update_func = GatedMLP(in_feats=node_input_dim, dims=[*atom_hidden_dims, num_atom_feats])
        self.node_out_func = layers.Dense(num_atom_feats, use_bias=False)

        if bond_hidden_dims is not None:
            self.edge_update_func = GatedMLP(in_feats=node_input_dim, dims=[*bond_hidden_dims, num_bond_feats])
        else:
            self.edge_update_func = None

        self.atom_norm = layers.LayerNormalization(axis=-1) if normalization == "layer" else None
        self.bond_norm = layers.LayerNormalization(axis=-1) if normalization == "layer" else None
        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, edge_index, atom_features, bond_features, bond_weights=None):
        src = ops.cast(edge_index[0], "int32")
        dst = ops.cast(edge_index[1], "int32")
        num_nodes = ops.shape(atom_features)[0]

        atom_i = ops.take(atom_features, src, axis=0)
        atom_j = ops.take(atom_features, dst, axis=0)
        inputs = ops.concatenate([atom_i, bond_features, atom_j], axis=-1)

        # Edge update
        if self.edge_update_func is not None:
            bond_update = self.edge_update_func(inputs)
            if bond_weights is not None:
                bond_update = bond_update * ops.expand_dims(bond_weights, axis=-1)
            new_bond_features = bond_features + bond_update
        else:
            new_bond_features = bond_features

        # Node update: scatter onto dst (neighbor atoms)
        messages = self.node_update_func(inputs)
        if bond_weights is not None:
            messages = messages * ops.expand_dims(bond_weights, axis=-1)
        node_update = scatter_add(messages, dst, num_segments=num_nodes)
        node_update = self.node_out_func(node_update)
        new_atom_features = atom_features + node_update

        if self.dropout is not None:
            new_atom_features = self.dropout(new_atom_features)
            new_bond_features = self.dropout(new_bond_features)

        if self.atom_norm is not None:
            new_atom_features = self.atom_norm(new_atom_features)
        if self.bond_norm is not None:
            new_bond_features = self.bond_norm(new_bond_features)

        return new_atom_features, new_bond_features


class CHGNetBondGraphBlock(layers.Layer):
    """Bond-graph (line-graph) convolution block for CHGNet.

    Example:
        ```python
        import numpy as np
        from k3_node.models import CHGNetBondGraphBlock

        line_edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])  # bond pairs forming 4 angles
        bond_feat = np.random.rand(8, 16).astype("float32")
        angle_feat = np.random.rand(4, 16).astype("float32")
        block = CHGNetBondGraphBlock(num_bond_feats=16, num_angle_feats=16, bond_hidden_dims=(16,))
        print(tuple(block(line_edge_index, bond_feat, angle_feat).shape))  # (8, 16): updated bond features
        ```
    """

    def __init__(
        self,
        num_bond_feats: int,
        num_angle_feats: int,
        bond_hidden_dims: Sequence[int] = (64,),
        activation: str = "swish",
        normalization: Optional[str] = "layer",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        lg_input_dim = 2 * num_bond_feats + num_angle_feats
        self.node_update_func = GatedMLP(in_feats=lg_input_dim, dims=[*bond_hidden_dims, num_bond_feats])
        self.node_out_func = layers.Dense(num_bond_feats, use_bias=False)
        self.bond_norm = layers.LayerNormalization(axis=-1) if normalization == "layer" else None
        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, line_edge_index, bond_features, angle_features, threebody_weights=None):
        src_bond = ops.cast(line_edge_index[0], "int32")
        dst_bond = ops.cast(line_edge_index[1], "int32")
        num_bonds = ops.shape(bond_features)[0]

        bond_i = ops.take(bond_features, src_bond, axis=0)
        bond_j = ops.take(bond_features, dst_bond, axis=0)
        inputs = ops.concatenate([bond_i, angle_features, bond_j], axis=-1)

        messages = self.node_update_func(inputs)
        if threebody_weights is not None:
            messages = messages * ops.expand_dims(threebody_weights, axis=-1)

        feat_update = scatter_add(messages, dst_bond, num_segments=num_bonds)
        feat_update = self.node_out_func(feat_update)
        new_bond_features = bond_features + feat_update

        if self.dropout is not None:
            new_bond_features = self.dropout(new_bond_features)
        if self.bond_norm is not None:
            new_bond_features = self.bond_norm(new_bond_features)

        return new_bond_features


class CHGNet(K3NodeHubMixin, keras.Model):
    """Crystal Hamiltonian Graph Neural Network (CHGNet) with charge and angular terms.

    Example:
        ```python
        import numpy as np
        from k3_node.models import CHGNet

        # A 4-atom structure: positions, bonds (listed in both directions) and atomic numbers
        structure = {
            "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.2, 0.8], [1.5, 1.5, 1.0]], dtype="float32"),
            "edge_index": np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]),
            "line_edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),  # bond pairs forming angles
            "node_type": np.array([6, 8, 1, 6]),  # atomic numbers
            "batch": np.zeros(4, dtype="int32"),  # all atoms belong to structure 0
            "state_attr": np.zeros((1, 2), dtype="float32"),  # global state features
        }

        model = CHGNet(dim_atom_embedding=16, dim_bond_embedding=16, dim_angle_embedding=16, num_blocks=2,
                       atom_conv_hidden_dims=(16,), bond_conv_hidden_dims=(16,))
        energy = model(structure)  # predicted property (e.g. energy) of the structure
        print(tuple(energy.shape))  # (1,)
        ```
    """

    def __init__(
        self,
        dim_atom_embedding: int = 64,
        dim_bond_embedding: int = 64,
        dim_angle_embedding: int = 64,
        cutoff: float = 6.0,
        threebody_cutoff: float = 3.0,
        cutoff_exponent: int = 5,
        max_n: int = 9,
        max_f: int = 4,
        num_blocks: int = 4,
        atom_conv_hidden_dims: Sequence[int] = (64,),
        bond_conv_hidden_dims: Sequence[int] = (64,),
        activation_type: str = "swish",
        normalization: Optional[str] = "layer",
        num_targets: int = 1,
        ntypes_node: int = 95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_atom_embedding = dim_atom_embedding
        self.dim_bond_embedding = dim_bond_embedding
        self.dim_angle_embedding = dim_angle_embedding
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.cutoff_exponent = cutoff_exponent
        self.max_n = max_n
        self.max_f = max_f
        self.num_blocks = num_blocks
        self.atom_conv_hidden_dims = list(atom_conv_hidden_dims)
        self.bond_conv_hidden_dims = list(bond_conv_hidden_dims)
        self.activation_type = activation_type
        self.normalization = normalization
        self.num_targets = num_targets
        self.ntypes_node = ntypes_node

        self.atom_embedding = layers.Embedding(ntypes_node, dim_atom_embedding)
        self.rbf = RadialBesselFunction(max_n=max_n, cutoff=cutoff, learnable=True)
        self.bond_embedding = layers.Dense(dim_bond_embedding, use_bias=True)

        self.angle_basis = FourierExpansion(max_f=max_f, interval=math.pi, learnable=True)
        self.angle_embedding = layers.Dense(dim_angle_embedding, use_bias=True)

        self.atom_blocks = []
        self.bond_blocks = []
        for _ in range(num_blocks):
            self.bond_blocks.append(
                CHGNetBondGraphBlock(
                    num_bond_feats=dim_bond_embedding,
                    num_angle_feats=dim_angle_embedding,
                    bond_hidden_dims=bond_conv_hidden_dims,
                    activation=activation_type,
                    normalization=normalization,
                )
            )
            self.atom_blocks.append(
                CHGNetAtomGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    atom_hidden_dims=atom_conv_hidden_dims,
                    bond_hidden_dims=bond_conv_hidden_dims,
                    activation=activation_type,
                    normalization=normalization,
                )
            )

        self.readout_pool = ReduceReadOut(op="sum")
        self.final_mlp = MLP([dim_atom_embedding, 64, 64, num_targets], activation=activation_type, activate_last=False)
        self.magmom_head = MLP([dim_atom_embedding, 32, 1], activation=activation_type, activate_last=False)

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            pos = inputs.get("pos")
            edge_index = inputs.get("edge_index")
            node_type = inputs.get("node_type", inputs.get("z"))
            line_edge_index = inputs.get("line_edge_index", None)
            pbc_offshift = inputs.get("pbc_offshift", None)
            batch = inputs.get("batch", None)
            num_graphs = inputs.get("num_graphs", None)
            state_attr = inputs.get("state_attr", None)
            return pos, edge_index, node_type, line_edge_index, pbc_offshift, batch, num_graphs, state_attr
        elif isinstance(inputs, (tuple, list)):
            pos = inputs[0]
            edge_index = inputs[1]
            node_type = inputs[2]
            line_edge_index = inputs[3] if len(inputs) > 3 else None
            pbc_offshift = inputs[4] if len(inputs) > 4 else None
            batch = inputs[5] if len(inputs) > 5 else None
            num_graphs = inputs[6] if len(inputs) > 6 else None
            state_attr = inputs[7] if len(inputs) > 7 else None
            return pos, edge_index, node_type, line_edge_index, pbc_offshift, batch, num_graphs, state_attr
        return inputs, None, None, None, None, None, None, None

    def call(
        self,
        inputs,
        edge_index=None,
        node_type=None,
        line_edge_index=None,
        pbc_offshift=None,
        batch=None,
        num_graphs=None,
        state_attr=None,
        return_magmom: bool = False,
    ):
        if edge_index is None:
            (
                pos,
                edge_index,
                node_type_in,
                line_edge_index_in,
                pbc_offshift_in,
                batch_in,
                num_graphs_in,
                state_attr_in,
            ) = self._unpack_inputs(inputs)
            if node_type is None:
                node_type = node_type_in
            if line_edge_index is None:
                line_edge_index = line_edge_index_in
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

        # 1. Geometry & initial embeddings
        _, bond_dists = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        atom_feats = self.atom_embedding(ops.cast(node_type, "int32"))
        rbf_feats = self.rbf(bond_dists)
        bond_weights = polynomial_cutoff(bond_dists, self.cutoff, exponent=self.cutoff_exponent)
        bond_feats = self.bond_embedding(rbf_feats)

        # 2. Line graph 3-body angles
        if line_edge_index is not None and ops.shape(line_edge_index)[1] > 0:
            thetas = compute_theta(pos, edge_index, line_edge_index, pbc_offshift)
            angle_feats = self.angle_embedding(self.angle_basis(thetas))
            src_bonds = ops.cast(line_edge_index[0], "int32")
            dst_bonds = ops.cast(line_edge_index[1], "int32")
            threebody_weights = (
                ops.take(bond_weights, src_bonds, axis=0) * ops.take(bond_weights, dst_bonds, axis=0)
            )
        else:
            angle_feats = None
            threebody_weights = None

        # 3. Convolution blocks
        for bond_block, atom_block in zip(self.bond_blocks, self.atom_blocks):
            if angle_feats is not None and line_edge_index is not None:
                bond_feats = bond_block(line_edge_index, bond_feats, angle_feats, threebody_weights=threebody_weights)
            atom_feats, bond_feats = atom_block(edge_index, atom_feats, bond_feats, bond_weights=bond_weights)

        # 4. Energy and magnetic moment prediction
        pooled = self.readout_pool(atom_feats, batch=batch, num_graphs=n_graphs)
        energy = ops.squeeze(self.final_mlp(pooled), axis=-1)

        if return_magmom:
            magmom = ops.squeeze(self.magmom_head(atom_feats), axis=-1)
            return energy, magmom
        return energy

