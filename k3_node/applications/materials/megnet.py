"""Multi-backend Keras 3 implementation of MEGNet."""

from __future__ import annotations

from typing import Sequence, Optional, Union, Tuple, Dict, Any
import keras
from keras import layers, ops
import numpy as np

from .core import MLP, EmbeddingBlock, SoftPlus2, get_activation, scatter_mean, infer_num_graphs
from .basis import BondExpansion, compute_pair_vector_and_distance
from .readout import Set2SetReadOut, EdgeSet2Set


class MEGNetGraphConv(layers.Layer):
    """MEGNet graph convolution layer: edge -> node -> state updates."""

    def __init__(
        self,
        edge_dims: Sequence[int],
        node_dims: Sequence[int],
        state_dims: Sequence[int],
        activation: str = "softplus2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_func = MLP(edge_dims, activation=activation, activate_last=True)
        self.node_func = MLP(node_dims, activation=activation, activate_last=True)
        self.state_func = MLP(state_dims, activation=activation, activate_last=True)

    def call(
        self,
        edge_index,
        edge_feat,
        node_feat,
        state_feat,
        batch=None,
        edge_batch=None,
        num_nodes: Optional[int] = None,
        num_graphs: Optional[int] = None,
    ):
        n_nodes = ops.shape(node_feat)[0]
        n_states = ops.shape(state_feat)[0]
        src = ops.clip(ops.cast(edge_index[0], "int32"), 0, ops.maximum(n_nodes - 1, 0))
        dst = ops.clip(ops.cast(edge_index[1], "int32"), 0, ops.maximum(n_nodes - 1, 0))

        # Broadcast state to nodes and edges
        if batch is not None:
            b_node = ops.clip(ops.cast(batch, "int32"), 0, ops.maximum(n_states - 1, 0))
            u_node = ops.take(state_feat, b_node, axis=0)
        else:
            num_n = ops.shape(node_feat)[0]
            u_node = ops.broadcast_to(state_feat, (num_n, ops.shape(state_feat)[-1]))

        if edge_batch is not None:
            b_edge = ops.clip(ops.cast(edge_batch, "int32"), 0, ops.maximum(n_states - 1, 0))
            u_edge = ops.take(state_feat, b_edge, axis=0)
        else:
            num_e = ops.shape(edge_feat)[0]
            u_edge = ops.broadcast_to(state_feat, (num_e, ops.shape(state_feat)[-1]))

        # 1. Edge update: concat(vi, vj, eij, u)
        vi = ops.take(node_feat, src, axis=0)
        vj = ops.take(node_feat, dst, axis=0)
        edge_inputs = ops.concatenate([vi, vj, edge_feat, u_edge], axis=-1)
        edge_feat_new = self.edge_func(edge_inputs)

        # 2. Node update: concat(vi, mean(eij), u)
        ve = scatter_mean(edge_feat_new, src, num_segments=num_nodes)
        node_inputs = ops.concatenate([node_feat, ve, u_node], axis=-1)
        node_feat_new = self.node_func(node_inputs)

        # 3. State update: concat(u, mean(e), mean(v))
        n_graphs = ops.shape(state_feat)[0]
        if edge_batch is not None:
            u_e_mean = scatter_mean(edge_feat_new, edge_batch, num_segments=n_graphs)
        else:
            u_e_mean = ops.mean(edge_feat_new, axis=0, keepdims=True)

        if batch is not None:
            u_v_mean = scatter_mean(node_feat_new, batch, num_segments=n_graphs)
        else:
            u_v_mean = ops.mean(node_feat_new, axis=0, keepdims=True)

        state_inputs = ops.concatenate([state_feat, u_e_mean, u_v_mean], axis=-1)
        state_feat_new = self.state_func(state_inputs)

        return edge_feat_new, node_feat_new, state_feat_new


class MEGNetBlock(layers.Layer):
    """MEGNet block: pre-MLPs, graph convolution, skip connections."""

    def __init__(
        self,
        dims: Sequence[int],
        conv_hiddens: Sequence[int],
        activation: str = "softplus2",
        dropout: float = 0.0,
        skip: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.skip = skip
        self.dims = list(dims)
        self.has_dense = len(self.dims) > 1
        conv_dim = self.dims[-1]
        out_dim = conv_hiddens[-1]

        if self.has_dense:
            self.edge_func = MLP(self.dims, activation=activation, activate_last=True)
            self.node_func = MLP(self.dims, activation=activation, activate_last=True)
            self.state_func = MLP(self.dims, activation=activation, activate_last=True)
        else:
            self.edge_func = lambda x: x
            self.node_func = lambda x: x
            self.state_func = lambda x: x

        edge_in = 2 * conv_dim + conv_dim + conv_dim
        node_in = out_dim + conv_dim + conv_dim
        state_in = out_dim + out_dim + conv_dim

        self.conv = MEGNetGraphConv(
            edge_dims=[edge_in, *conv_hiddens],
            node_dims=[node_in, *conv_hiddens],
            state_dims=[state_in, *conv_hiddens],
            activation=activation,
        )
        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

    def call(
        self,
        edge_index,
        edge_feat,
        node_feat,
        state_feat,
        batch=None,
        edge_batch=None,
        num_nodes: Optional[int] = None,
        num_graphs: Optional[int] = None,
    ):
        in_edge, in_node, in_state = edge_feat, node_feat, state_feat

        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_feat = self.state_func(state_feat)

        edge_feat, node_feat, state_feat = self.conv(
            edge_index, edge_feat, node_feat, state_feat,
            batch=batch, edge_batch=edge_batch, num_nodes=num_nodes, num_graphs=num_graphs
        )

        if self.dropout is not None:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            state_feat = self.dropout(state_feat)

        if self.skip:
            edge_feat = edge_feat + in_edge
            node_feat = node_feat + in_node
            state_feat = state_feat + in_state

        return edge_feat, node_feat, state_feat


class MEGNet(keras.Model):
    """MEGNet materials graph network supporting TensorFlow, PyTorch, and JAX."""

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: Optional[int] = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: Sequence[int] = (64, 32),
        hidden_layer_sizes_conv: Sequence[int] = (64, 64, 32),
        hidden_layer_sizes_output: Sequence[int] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float = 0.0,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        ntypes_node: int = 95,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.is_classification = is_classification
        self.include_state = include_state

        self.bond_expansion = BondExpansion(
            rbf_type="Gaussian",
            initial=0.0,
            final=cutoff + 1.0,
            num_centers=dim_edge_embedding,
            width=gauss_width,
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=ntypes_node,
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation_type,
        )

        self.edge_encoder = MLP(edge_dims, activation=activation_type, activate_last=True)
        self.node_encoder = MLP(node_dims, activation=activation_type, activate_last=True)
        self.state_encoder = MLP(state_dims, activation=activation_type, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]

        self.blocks = []
        for i in range(nblocks):
            dims = [dim_blocks_in] if i == 0 else [dim_blocks_out, *hidden_layer_sizes_input]
            self.blocks.append(
                MEGNetBlock(
                    dims=dims,
                    conv_hiddens=hidden_layer_sizes_conv,
                    activation=activation_type,
                    dropout=dropout,
                    skip=True,
                )
            )

        self.node_s2s = Set2SetReadOut(dim_blocks_out, processing_steps=niters_set2set, num_layers=nlayers_set2set)
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, n_iters=niters_set2set, n_layers=nlayers_set2set)

        self.output_proj = MLP(
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation_type,
            activate_last=False,
        )

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            pos = inputs.get("pos")
            edge_index = inputs.get("edge_index")
            node_type = inputs.get("node_type", inputs.get("z"))
            state_attr = inputs.get("state_attr", None)
            pbc_offshift = inputs.get("pbc_offshift", None)
            batch = inputs.get("batch", None)
            num_graphs = inputs.get("num_graphs", None)
            return pos, edge_index, node_type, state_attr, pbc_offshift, batch, num_graphs
        elif isinstance(inputs, (tuple, list)):
            pos = inputs[0]
            edge_index = inputs[1]
            node_type = inputs[2]
            state_attr = inputs[3] if len(inputs) > 3 else None
            pbc_offshift = inputs[4] if len(inputs) > 4 else None
            batch = inputs[5] if len(inputs) > 5 else None
            num_graphs = inputs[6] if len(inputs) > 6 else None
            return pos, edge_index, node_type, state_attr, pbc_offshift, batch, num_graphs
        return inputs, None, None, None, None, None, None

    def call(self, inputs, edge_index=None, node_type=None, state_attr=None, pbc_offshift=None, batch=None, num_graphs=None):
        if edge_index is None:
            (
                pos,
                edge_index,
                node_type_in,
                state_attr_in,
                pbc_offshift_in,
                batch_in,
                num_graphs_in,
            ) = self._unpack_inputs(inputs)
            if node_type is None:
                node_type = node_type_in
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
        src_safe = ops.clip(src, 0, ops.maximum(ops.shape(batch)[0] - 1, 0))
        edge_batch = ops.take(batch, src_safe, axis=0)

        # 1. Expand pairwise distances
        _, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        edge_attr = self.bond_expansion(bond_dist)

        # 2. State attributes
        if state_attr is None:
            state_attr = ops.zeros((n_graphs, 2), dtype="float32")

        # 3. Embeddings & encoders
        node_feat, edge_feat, state_feat = self.embedding(node_type, edge_attr, state_attr)
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        # 4. Convolution blocks
        for block in self.blocks:
            edge_feat, node_feat, state_feat = block(
                edge_index, edge_feat, node_feat, state_feat,
                batch=batch, edge_batch=edge_batch, num_nodes=num_nodes, num_graphs=n_graphs
            )

        # 5. Readout pooling
        node_vec = self.node_s2s(node_feat, batch=batch, num_graphs=n_graphs)
        edge_vec = self.edge_s2s(edge_feat, edge_batch=edge_batch, num_graphs=n_graphs)
        state_vec = ops.reshape(state_feat, (n_graphs, -1))

        vec = ops.concatenate([node_vec, edge_vec, state_vec], axis=-1)
        output = self.output_proj(vec)
        if self.is_classification:
            output = ops.sigmoid(output)

        return ops.squeeze(output, axis=-1)

