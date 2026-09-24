"""Core layers and mathematical primitives for materials models."""

from __future__ import annotations

import math
from typing import Sequence, Callable, Optional, Union
import keras
from keras import layers, ops
import numpy as np


class SoftPlus2(layers.Layer):
    """SoftPlus2 activation: log(exp(x) + 1) - log(2). Zero at the origin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shift = float(math.log(2.0))

    def call(self, x):
        return ops.softplus(x) - self.shift


class SoftExponential(layers.Layer):
    """Soft exponential activation with learnable alpha."""

    def __init__(self, alpha: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = float(alpha)
        self._eps = 1e-6

    def build(self, input_shape=None):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=keras.initializers.Constant(self.init_alpha),
            trainable=True,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, x):
        alpha = self.alpha
        near_zero = ops.abs(alpha) < self._eps
        safe_alpha = ops.where(near_zero, 1.0, alpha)

        neg_log_arg = ops.where(alpha < 0.0, 1.0 - alpha * (x + alpha), ops.ones_like(x))
        neg_log_arg = ops.maximum(neg_log_arg, self._eps)
        neg = -ops.log(neg_log_arg) / safe_alpha
        pos = ops.expm1(safe_alpha * x) / safe_alpha + safe_alpha

        out = ops.where(alpha < 0.0, neg, pos)
        return ops.where(near_zero, x, out)


def get_activation(act: Union[str, Callable, layers.Layer, None]):
    """Resolve activation specification into a callable or Keras layer."""
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        act_lower = act.lower()
        if act_lower in ("swish", "silu"):
            return ops.silu
        elif act_lower == "softplus2":
            return SoftPlus2()
        elif act_lower == "softexp":
            return SoftExponential()
        elif act_lower == "softplus":
            return ops.softplus
        elif act_lower == "tanh":
            return ops.tanh
        elif act_lower == "sigmoid":
            return ops.sigmoid
        elif act_lower == "relu":
            return ops.relu
        elif act_lower in ("identity", "linear", "none"):
            return lambda x: x
        return keras.activations.get(act)
    if isinstance(act, layers.Layer):
        return act
    return act


class MLP(layers.Layer):
    """Multi-layer perceptron compatible with multi-backend Keras 3."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: Union[str, Callable, layers.Layer, None] = "swish",
        activate_last: bool = False,
        bias_last: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = list(dims)
        self.activate_last = activate_last
        self.bias_last = bias_last
        self.use_bias = use_bias
        self.act_fn = get_activation(activation)

        self.dense_layers = []
        for i in range(len(self.dims) - 1):
            is_last = (i == len(self.dims) - 2)
            bias = self.bias_last if is_last else self.use_bias
            dense = layers.Dense(self.dims[i + 1], use_bias=bias)
            if self.dims[i] is not None:
                dense.build((None, self.dims[i]))
            self.dense_layers.append(dense)

    def call(self, x):
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            is_last = (i == len(self.dense_layers) - 1)
            if not is_last or self.activate_last:
                x = self.act_fn(x)
        return x


class GatedMLP(layers.Layer):
    """Gated multi-layer perceptron: layer(x) * sigmoid(gate(x))."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activate_last: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self.activate_last = activate_last
        self.use_bias = use_bias

        self.val_layers = []
        self.gate_layers = []
        for i in range(len(self.dims) - 1):
            out_dim = self.dims[i + 1]
            val = layers.Dense(out_dim, use_bias=use_bias)
            gate = layers.Dense(out_dim, use_bias=use_bias)
            if self.dims[i] is not None:
                val.build((None, self.dims[i]))
                gate.build((None, self.dims[i]))
            self.val_layers.append(val)
            self.gate_layers.append(gate)

    def call(self, x):
        h_val = x
        h_gate = x
        for i in range(len(self.val_layers)):
            is_last = (i == len(self.val_layers) - 1)
            h_val = self.val_layers[i](h_val)
            h_gate = self.gate_layers[i](h_gate)
            if not is_last:
                h_val = ops.silu(h_val)
                h_gate = ops.silu(h_gate)
            else:
                if self.activate_last:
                    h_val = ops.silu(h_val)
                h_gate = ops.sigmoid(h_gate)
        return h_val * h_gate


class EmbeddingBlock(layers.Layer):
    """Embeddings for nodes (atoms), edges (bonds), and global states."""

    def __init__(
        self,
        degree_rbf: int,
        dim_node_embedding: int,
        dim_edge_embedding: Optional[int] = None,
        dim_state_embedding: Optional[int] = None,
        dim_state_feats: Optional[int] = None,
        ntypes_node: Optional[int] = None,
        ntypes_state: Optional[int] = None,
        include_state: bool = False,
        activation: Union[str, Callable, None] = "swish",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.degree_rbf = degree_rbf
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_embedding = dim_state_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.ntypes_state = ntypes_state
        self.include_state = include_state

        if ntypes_node is not None:
            self.layer_node_embedding = layers.Embedding(ntypes_node, dim_node_embedding)
        else:
            self.layer_node_embedding = layers.Dense(dim_node_embedding, use_bias=False)

        if dim_edge_embedding is not None:
            self.layer_edge_embedding = MLP([degree_rbf, dim_edge_embedding], activation=activation, activate_last=True)
        else:
            self.layer_edge_embedding = None

        if include_state:
            if ntypes_state is not None and dim_state_embedding is not None:
                self.layer_state_embedding = layers.Embedding(ntypes_state, dim_state_embedding)
            elif dim_state_feats is not None:
                self.layer_state_embedding = layers.Dense(dim_state_feats, use_bias=False)
            elif dim_state_embedding is not None:
                self.layer_state_embedding = layers.Dense(dim_state_embedding, use_bias=False)
            else:
                self.layer_state_embedding = None
        else:
            self.layer_state_embedding = None

    def call(self, node_attr, edge_attr, state_attr=None):
        if self.ntypes_node is not None:
            node_idx = ops.clip(ops.cast(node_attr, "int32"), 0, self.ntypes_node - 1)
            node_feat = self.layer_node_embedding(node_idx)
        else:
            node_feat = self.layer_node_embedding(node_attr)

        edge_feat = self.layer_edge_embedding(edge_attr) if self.layer_edge_embedding is not None else edge_attr

        state_feat = None
        if self.include_state and state_attr is not None and self.layer_state_embedding is not None:
            if self.ntypes_state is not None:
                state_idx = ops.clip(ops.cast(state_attr, "int32"), 0, self.ntypes_state - 1)
                state_feat = self.layer_state_embedding(state_idx)
            else:
                state_feat = self.layer_state_embedding(state_attr)
        elif self.include_state and state_attr is not None:
            state_feat = state_attr

        return node_feat, edge_feat, state_feat


# --- Tensor math utilities for TensorNet ---

def vector_to_skewtensor(vector):
    """Create skew-symmetric 3x3 tensor from a 3D vector.

    ```
    [0, -v_z, v_y]
    [v_z, 0, -v_x]
    [-v_y, v_x, 0]
    ```
    """
    vector = ops.convert_to_tensor(vector)
    vx = vector[..., 0]
    vy = vector[..., 1]
    vz = vector[..., 2]
    zero = ops.zeros_like(vx)

    row0 = ops.stack([zero, -vz, vy], axis=-1)
    row1 = ops.stack([vz, zero, -vx], axis=-1)
    row2 = ops.stack([-vy, vx, zero], axis=-1)
    return ops.stack([row0, row1, row2], axis=-2)


def vector_to_symtensor(vector):
    """Create symmetric traceless tensor from outer product of 3D vector with itself."""
    vector = ops.convert_to_tensor(vector)
    v_col = ops.expand_dims(vector, axis=-1)
    v_row = ops.expand_dims(vector, axis=-2)
    outer = ops.matmul(v_col, v_row)
    # trace = outer[..., 0, 0] + outer[..., 1, 1] + outer[..., 2, 2]
    trace = outer[..., 0, 0] + outer[..., 1, 1] + outer[..., 2, 2]
    eye3 = ops.eye(3, dtype=outer.dtype)
    scalars = ops.expand_dims(ops.expand_dims(trace / 3.0, axis=-1), axis=-1) * eye3
    sym = 0.5 * (outer + ops.transpose(outer, axes=[*range(len(outer.shape) - 2), -1, -2]))
    return sym - scalars


def decompose_tensor(tensor):
    """Decompose 3x3 Cartesian tensor into scalar (I), skew-symmetric (A), and symmetric traceless (S)."""
    tensor = ops.convert_to_tensor(tensor)
    trace = tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]
    eye3 = ops.eye(3, dtype=tensor.dtype)
    scalars = ops.expand_dims(ops.expand_dims(trace / 3.0, axis=-1), axis=-1) * eye3
    transposed = ops.transpose(tensor, axes=[*range(len(tensor.shape) - 2), -1, -2])
    skew = 0.5 * (tensor - transposed)
    sym = 0.5 * (tensor + transposed)
    traceless = sym - scalars
    return scalars, skew, traceless


def new_radial_tensor(scalars, skew, traceless, f_I, f_A, f_S):
    """Multiply irreducible tensor components by radial invariant features."""
    scalars_out = ops.expand_dims(ops.expand_dims(f_I, axis=-1), axis=-1) * scalars
    skew_out = ops.expand_dims(ops.expand_dims(f_A, axis=-1), axis=-1) * skew
    traceless_out = ops.expand_dims(ops.expand_dims(f_S, axis=-1), axis=-1) * traceless
    return scalars_out, skew_out, traceless_out


def tensor_norm(tensor):
    """Computes Frobenius norm squared across last two dimensions (3, 3)."""
    return ops.sum(tensor ** 2, axis=(-2, -1))


def scatter_add(x, index, num_segments: int):
    """Scatter sum x elements into segments indicated by index."""
    index = ops.cast(index, "int32")
    return ops.segment_sum(x, index, num_segments=num_segments)


def scatter_mean(x, index, num_segments: int):
    """Scatter mean x elements into segments indicated by index."""
    index = ops.cast(index, "int32")
    sums = ops.segment_sum(x, index, num_segments=num_segments)
    ones = ops.ones_like(x[..., :1])
    counts = ops.segment_sum(ones, index, num_segments=num_segments)
    counts = ops.maximum(counts, 1.0)
    return sums / counts


def infer_num_graphs(batch=None, num_graphs=None, state_attr=None):
    """Infer the number of graphs in a batch across backends (JAX, PyTorch, TF)."""
    if num_graphs is not None:
        return num_graphs
    if state_attr is not None:
        shape = ops.shape(state_attr)
        if len(shape) == 1:
            return 1
        return shape[0]
    if batch is None:
        return 1
    shape = ops.shape(batch)
    if len(shape) > 0 and shape[0] == 0:
        return 0
    try:
        val = ops.convert_to_numpy(ops.max(batch))
        return int(val) + 1
    except Exception:
        pass
    try:
        if hasattr(batch, "__getitem__"):
            val = batch[-1]
            if hasattr(val, "item"):
                return int(val.item()) + 1
            return int(val) + 1
    except Exception:
        pass
    try:
        return ops.cast(ops.max(batch) + 1, "int32")
    except Exception:
        return 1

