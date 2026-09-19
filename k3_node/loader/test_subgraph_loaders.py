import numpy as np
try:
    import torch
except ImportError:
    torch = None

from k3_node.data import Data
from k3_node.loader import (
    ClusterData,
    ClusterLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborSampler,
    ShaDowKHopSampler,
)


def get_graph(num_nodes=12):
    # Circular ring graph
    row = np.arange(num_nodes)
    col = (row + 1) % num_nodes
    edge_index = np.stack([row, col], axis=0).astype(np.int64)
    x = np.random.randn(num_nodes, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_nodes,)).astype(np.int64)

    if torch is not None:
        edge_index = torch.from_numpy(edge_index)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

    return Data(x=x, edge_index=edge_index, y=y)


def test_cluster_loader():
    data = get_graph(num_nodes=12)
    cluster_data = ClusterData(data, num_parts=3)
    assert len(cluster_data) == 3

    sub0 = cluster_data[0]
    assert isinstance(sub0, Data)
    assert sub0.num_nodes > 0

    loader = ClusterLoader(cluster_data, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2  # 3 parts with batch_size 2 => 2 batches


def test_graph_saint():
    data = get_graph(num_nodes=12)

    # Node sampler
    node_sampler = GraphSAINTNodeSampler(data, batch_size=4, num_steps=2, sample_coverage=1)
    batch = next(iter(node_sampler))
    assert isinstance(batch, Data)
    assert hasattr(batch, 'node_norm')

    # Edge sampler
    edge_sampler = GraphSAINTEdgeSampler(data, batch_size=4, num_steps=2)
    batch = next(iter(edge_sampler))
    assert isinstance(batch, Data)

    # Random walk sampler
    rw_sampler = GraphSAINTRandomWalkSampler(data, batch_size=4, walk_length=2, num_steps=2)
    batch = next(iter(rw_sampler))
    assert isinstance(batch, Data)


def test_shadow_k_hop_sampler():
    data = get_graph(num_nodes=12)
    loader = ShaDowKHopSampler(data, depth=2, num_neighbors=2, batch_size=3)

    batch = next(iter(loader))
    assert batch.num_graphs == 3
    assert hasattr(batch, 'root_n_id')


def test_legacy_neighbor_sampler():
    data = get_graph(num_nodes=12)
    loader = NeighborSampler(data.edge_index, sizes=[2, 2], batch_size=3)

    batch_size, n_id, adjs = next(iter(loader))
    assert batch_size == 3
    assert len(n_id) >= 3
    assert len(adjs) == 2
    for adj in adjs:
        assert hasattr(adj, 'edge_index')
        assert hasattr(adj, 'size')

