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


def structured_negative_sampling(edge_index, num_nodes=None, contains_neg_self_loops=True):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.
    """
    edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)
    if num_nodes is None:
        num_nodes = int(edge_index_np.max()) + 1 if edge_index_np.size > 0 else 0

    row, col = edge_index_np[0], edge_index_np[1]
    pos_idx = set((row * num_nodes + col).tolist())
    if not contains_neg_self_loops:
        loop_idx = np.arange(num_nodes) * (num_nodes + 1)
        pos_idx.update(loop_idx.tolist())

    k_list = []
    for r in row:
        for _ in range(100):
            cand = int(np.random.randint(0, max(num_nodes, 1)))
            if (r * num_nodes + cand) not in pos_idx:
                k_list.append(cand)
                break
        else:
            k_list.append(0)

    k_tensor = ops.convert_to_tensor(np.array(k_list, dtype=np.int64), dtype="int64")
    return edge_index[0], edge_index[1], k_tensor

