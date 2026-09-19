import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

from k3_node.data import Data
from k3_node.loader import (
    CachedLoader,
    DataLoader,
    DynamicBatchSampler,
    ImbalancedSampler,
    PrefetchLoader,
    ZipLoader,
)


def create_dummy_data(num_nodes=4, num_features=8, label=0):
    x = np.random.randn(num_nodes, num_features).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    y = np.array([label], dtype=np.int64)
    if torch is not None:
        x = torch.from_numpy(x)
        edge_index = torch.from_numpy(edge_index)
        y = torch.from_numpy(y)
    return Data(x=x, edge_index=edge_index, y=y)


def test_dynamic_batch_sampler():
    dataset = [create_dummy_data(num_nodes=i + 2) for i in range(5)]
    sampler = DynamicBatchSampler(dataset, max_num=15, mode='node')

    loader = DataLoader(dataset, batch_sampler=sampler)
    count = 0
    for batch in loader:
        assert batch.num_nodes <= 15
        count += 1
    assert count > 0

    with pytest.raises(ValueError, match="length of 'DynamicBatchSampler'"):
        len(sampler)
    assert len(DynamicBatchSampler(dataset, max_num=15, num_steps=2)) == 2


def test_imbalanced_sampler():
    # 8 samples with label 0, 2 samples with label 1
    dataset = [create_dummy_data(label=0) for _ in range(8)] + [create_dummy_data(label=1) for _ in range(2)]
    sampler = ImbalancedSampler(dataset, num_samples=10)

    sampled_indices = list(sampler)
    assert len(sampled_indices) == 10

    loader = DataLoader(dataset, batch_size=5, sampler=sampler)
    batches = list(loader)
    assert len(batches) == 2


def test_zip_loader():
    from k3_node.loader import NeighborLoader
    data = create_dummy_data(num_nodes=10)

    loader1 = NeighborLoader(data, num_neighbors=[2], input_nodes=[0, 1, 2, 3])
    loader2 = NeighborLoader(data, num_neighbors=[2], input_nodes=[4, 5, 6, 7])

    zip_loader = ZipLoader([loader1, loader2], batch_size=2)
    for batch1, batch2 in zip_loader:
        assert batch1.batch_size == 2
        assert batch2.batch_size == 2


def test_cached_loader():
    dataset = [create_dummy_data(num_nodes=3) for _ in range(4)]
    loader = DataLoader(dataset, batch_size=2)

    cached_loader = CachedLoader(loader)
    epoch1 = list(cached_loader)
    epoch2 = list(cached_loader)

    assert len(epoch1) == 2
    assert len(epoch2) == 2
    assert len(cached_loader) == 2

    cached_loader.clear()
    assert len(cached_loader._cache) == 0


def test_prefetch_loader():
    dataset = [create_dummy_data(num_nodes=3) for _ in range(4)]
    loader = DataLoader(dataset, batch_size=2)

    prefetch = PrefetchLoader(loader)
    batches = list(prefetch)
    assert len(batches) == 2
    assert len(prefetch) == 2
