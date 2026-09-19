import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from k3_node.data import Data, HeteroData, TemporalData
from k3_node.loader import (
    DataListLoader,
    DataLoader,
    DenseDataLoader,
    RandomNodeLoader,
    TemporalDataLoader,
)


def create_dummy_data(num_nodes=4, num_features=8):
    x = np.random.randn(num_nodes, num_features).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    if torch is not None:
        x = torch.from_numpy(x)
        edge_index = torch.from_numpy(edge_index)
        y = torch.from_numpy(y)
    return Data(x=x, edge_index=edge_index, y=y)


def test_data_loader_basic():
    dataset = [create_dummy_data(num_nodes=i + 2) for i in range(4)]
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2

    batch0 = batches[0]
    assert batch0.num_graphs == 2
    assert hasattr(batch0, 'batch')
    assert hasattr(batch0, 'ptr')
    assert batch0.num_nodes == dataset[0].num_nodes + dataset[1].num_nodes


def test_data_loader_follow_batch_and_exclude():
    dataset = [create_dummy_data(num_nodes=3) for _ in range(3)]
    loader = DataLoader(dataset, batch_size=2, follow_batch=['y'], exclude_keys=['edge_index'])

    batch = next(iter(loader))
    assert 'edge_index' not in batch
    assert hasattr(batch, 'y_batch')


def test_data_list_loader():
    dataset = [create_dummy_data(num_nodes=3) for _ in range(4)]
    loader = DataListLoader(dataset, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2
    assert isinstance(batches[0], list)
    assert len(batches[0]) == 2
    assert isinstance(batches[0][0], Data)


def test_dense_data_loader():
    def create_dense_graph(num_nodes=4, num_features=6):
        x = np.random.randn(num_nodes, num_features).astype(np.float32)
        adj = np.random.randn(num_nodes, num_nodes).astype(np.float32)
        if torch is not None:
            x = torch.from_numpy(x)
            adj = torch.from_numpy(adj)
        return Data(x=x, adj=adj)

    dataset = [create_dense_graph() for _ in range(4)]
    loader = DenseDataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert batch.x.shape == (2, 4, 6)
    assert batch.adj.shape == (2, 4, 4)


def test_temporal_data_loader():
    src = np.array([0, 1, 0, 2, 1, 3], dtype=np.int64)
    dst = np.array([1, 2, 2, 3, 3, 0], dtype=np.int64)
    t = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    msg = np.random.randn(6, 4).astype(np.float32)

    if torch is not None:
        src = torch.from_numpy(src)
        dst = torch.from_numpy(dst)
        t = torch.from_numpy(t)
        msg = torch.from_numpy(msg)

    data = TemporalData(src=src, dst=dst, t=t, msg=msg)
    loader = TemporalDataLoader(data, batch_size=3, neg_sampling_ratio=1.0)

    batches = list(loader)
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert hasattr(batches[0], 'neg_dst')
    assert hasattr(batches[0], 'n_id')


def test_random_node_loader():
    data = create_dummy_data(num_nodes=10)
    loader = RandomNodeLoader(data, num_parts=2)

    parts = list(loader)
    assert len(parts) == 2
    for part in parts:
        assert isinstance(part, Data)
        assert part.num_nodes <= 10
        assert hasattr(part, 'x')

