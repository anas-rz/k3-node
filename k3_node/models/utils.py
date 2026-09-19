import math

import numpy as np
from keras import ops


def reset(value):
    r"""Resets learnable parameters, mirroring `torch_geometric.nn.inits.reset`.

    Calls `value.reset_parameters()` when available; otherwise recurses into
    Keras sublayers. Silently no-ops for values with neither (e.g. plain
    functions or layers whose weights are already initialized eagerly).
    """
    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    elif hasattr(value, "_layers"):
        for child in value._layers:
            reset(child)


def uniform_(size: int, value):
    r"""In-place-style uniform init over `[-1/sqrt(size), 1/sqrt(size)]`,
    mirroring `torch_geometric.nn.inits.uniform`. `value` must be a Keras
    Variable (e.g. a layer's weight).
    """
    if value is None:
        return
    bound = 1.0 / math.sqrt(size)
    value.assign(ops.convert_to_tensor(np.random.uniform(-bound, bound, ops.shape(value)), dtype=value.dtype))


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None):
    r"""Samples random negative edges of a graph given by `edge_index`.

    A simplified, framework-agnostic reimplementation of
    `torch_geometric.utils.negative_sampling` (non-bipartite, sparse-method
    case only), used by `k3_node.models.GAE.recon_loss`.
    """
    edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)

    if num_nodes is None:
        num_nodes = int(edge_index_np.max()) + 1 if edge_index_np.size > 0 else 0
    if num_neg_samples is None:
        num_neg_samples = edge_index_np.shape[1]

    population = num_nodes * num_nodes
    pos_idx = set((edge_index_np[0] * num_nodes + edge_index_np[1]).tolist())

    neg_idx = []
    seen = set()
    for _ in range(20):
        if len(neg_idx) >= num_neg_samples:
            break
        sample_size = int(1.5 * (num_neg_samples - len(neg_idx))) + 1
        rnd = np.random.randint(0, max(population, 1), size=sample_size)
        for v in rnd:
            v = int(v)
            if v not in pos_idx and v not in seen:
                seen.add(v)
                neg_idx.append(v)
                if len(neg_idx) >= num_neg_samples:
                    break

    neg_idx = np.array(neg_idx[:num_neg_samples], dtype=np.int64)
    row = neg_idx // max(num_nodes, 1)
    col = neg_idx % max(num_nodes, 1)
    return ops.convert_to_tensor(np.stack([row, col], axis=0), dtype="int64")
