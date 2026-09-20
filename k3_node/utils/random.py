from typing import List, Union
import numpy as np
from keras import ops


def erdos_renyi_graph(
    num_nodes: int,
    edge_prob: float,
    directed: bool = False,
):
    r"""Returns the edge_index of a random Erdos-Renyi graph."""
    from k3_node.transforms.utils import to_undirected
    if directed:
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.rand() < edge_prob:
                    rows.append(i)
                    cols.append(j)
        edge_index = (
            np.array([rows, cols], dtype=np.int64)
            if len(rows) > 0
            else np.zeros((2, 0), dtype=np.int64)
        )
    else:
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_prob:
                    rows.append(i)
                    cols.append(j)
        if len(rows) > 0:
            edge_index = np.array([rows, cols], dtype=np.int64)
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

    return ops.convert_to_tensor(edge_index, dtype="int64")


def stochastic_blockmodel_graph(
    block_sizes: Union[List[int], np.ndarray],
    edge_probs: Union[List[List[float]], np.ndarray],
    directed: bool = False,
):
    r"""Returns the edge_index of a stochastic blockmodel graph."""
    from k3_node.transforms.utils import to_undirected
    block_sizes_np = np.array(block_sizes, dtype=np.int64)
    edge_probs_np = np.array(edge_probs, dtype=np.float32)
    node_idx = np.concatenate([np.full(b, i, dtype=np.int64) for i, b in enumerate(block_sizes_np)])
    num_nodes = len(node_idx)

    rows, cols = [], []
    if directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    prob = edge_probs_np[node_idx[i], node_idx[j]]
                    if np.random.rand() < prob:
                        rows.append(i)
                        cols.append(j)
        edge_index = (
            np.array([rows, cols], dtype=np.int64)
            if len(rows) > 0
            else np.zeros((2, 0), dtype=np.int64)
        )
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                prob = edge_probs_np[node_idx[i], node_idx[j]]
                if np.random.rand() < prob:
                    rows.append(i)
                    cols.append(j)
        if len(rows) > 0:
            edge_index = np.array([rows, cols], dtype=np.int64)
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

    return ops.convert_to_tensor(edge_index, dtype="int64")


def barabasi_albert_graph(num_nodes: int, num_edges: int):
    r"""Returns the edge_index of a Barabasi-Albert preferential attachment model."""
    assert 0 < num_edges < num_nodes
    from k3_node.layers.conv.utils import remove_self_loops
    from k3_node.transforms.utils import to_undirected

    row = list(range(num_edges))
    col = list(np.random.permutation(num_edges))

    for i in range(num_edges, num_nodes):
        row.extend([i] * num_edges)
        degree_pool = row + col
        choice = np.random.choice(degree_pool, size=num_edges, replace=True)
        col.extend(choice.tolist())

    edge_index = np.array([row, col], dtype=np.int64)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return ops.convert_to_tensor(edge_index, dtype="int64")

