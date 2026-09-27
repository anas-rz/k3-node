import numpy as np
import pytest

from k3_node.loader.sampler_utils import FastGraph, sample_neighbors_homo


def _reference_full_sampling(edge_index, seeds, num_hops, subgraph_type):
    """The original per-node Python algorithm, restricted to k=-1 where it is deterministic."""
    graph = FastGraph(edge_index)
    nodes, visited = [], {}
    for s in seeds:
        if int(s) not in visited:
            visited[int(s)] = len(nodes)
            nodes.append(int(s))
    frontier, sampled = list(nodes), []
    for _ in range(num_hops):
        next_frontier = []
        for target in frontier:
            srcs, e_ids = graph.get_neighbors(target)
            for s, e in zip(srcs, e_ids):
                sampled.append((int(s), target, int(e)))
                if int(s) not in visited:
                    visited[int(s)] = len(nodes)
                    nodes.append(int(s))
                    next_frontier.append(int(s))
        frontier = next_frontier
    if subgraph_type == "induced":
        edges = [(visited[u], visited[v], e) for e, (u, v) in enumerate(edge_index.T) if u in visited and v in visited]
    elif subgraph_type == "bidirectional":
        edges = [x for u, v, e in sampled for x in ((visited[u], visited[v], e), (visited[v], visited[u], e))]
    else:
        edges = [(visited[u], visited[v], e) for u, v, e in sampled]
    edges = np.array(edges, dtype=np.int64).reshape(-1, 3)
    return np.array(nodes), edges[:, 0], edges[:, 1], edges[:, 2]


def _random_graph(num_nodes=40, num_edges=160, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_nodes, (2, num_edges)).astype(np.int64)


@pytest.mark.parametrize("subgraph_type", ["directional", "bidirectional", "induced"])
def test_full_neighborhood_matches_reference(subgraph_type):
    edge_index = _random_graph()
    seeds = np.array([3, 7, 3, 11])
    expected = _reference_full_sampling(edge_index, seeds, 2, subgraph_type)
    nodes, row, col, edge, _, _ = sample_neighbors_homo(edge_index, seeds, [-1, -1], subgraph_type=subgraph_type)
    for got, want in zip((nodes, row, col, edge), expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("replace", [False, True])
def test_sampled_neighborhood_properties(replace):
    edge_index = _random_graph()
    graph = FastGraph(edge_index)
    seeds = np.array([0, 5, 9])
    nodes, row, col, edge, n_counts, e_counts = sample_neighbors_homo(
        edge_index, seeds, [3, 2], replace=replace, graph=graph
    )
    assert len(np.unique(nodes)) == len(nodes)
    np.testing.assert_array_equal(nodes[:3], seeds)
    assert sum(n_counts) == len(nodes) and sum(e_counts) == len(edge)
    # Every sampled edge is a real edge, with endpoints mapped to their local ids.
    np.testing.assert_array_equal(nodes[row], edge_index[0, edge])
    np.testing.assert_array_equal(nodes[col], edge_index[1, edge])
    # First hop: each seed gets min(3, in-degree) edges, distinct when sampling without replacement.
    in_degree = np.bincount(edge_index[1], minlength=40)
    first_hop = slice(0, e_counts[0])
    for i, s in enumerate(seeds):
        picked = edge[first_hop][col[first_hop] == i]
        assert len(picked) == min(3, in_degree[s])
        if not replace:
            assert len(np.unique(picked)) == len(picked)
    # The reusable id map is left clean for the next batch.
    assert (graph.local_map(40) == -1).all()


def test_disjoint_and_out_of_range_seeds():
    edge_index = np.array([[1, 2], [0, 0]])
    nodes, row, col, edge, _, _ = sample_neighbors_homo(edge_index, np.array([0, 5]), [-1], num_nodes=3, disjoint=True)
    np.testing.assert_array_equal(nodes, [0, 5, 1, 2])
    np.testing.assert_array_equal(edge, [0, 1])
