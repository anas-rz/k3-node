import time
import numpy as np
import pytest
import torch

import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader

import k3_node.data as k3_data
import k3_node.loader as k3_loader


def create_pyg_data(num_nodes=5, num_features=8):
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 0][:num_nodes], dtype=torch.long)
    return pyg_data.Data(x=x, edge_index=edge_index, y=y)


def create_k3_data(num_nodes=5, num_features=8):
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 0][:num_nodes], dtype=torch.long)
    return k3_data.Data(x=x, edge_index=edge_index, y=y)


def test_reference_dataloader_parity():
    # Identical graphs in PyG and k3_node
    torch.manual_seed(42)
    pyg_dataset = [create_pyg_data(num_nodes=4 + i) for i in range(8)]
    torch.manual_seed(42)
    k3_dataset = [create_k3_data(num_nodes=4 + i) for i in range(8)]

    pyg_dl = pyg_loader.DataLoader(pyg_dataset, batch_size=4, shuffle=False)
    k3_dl = k3_loader.DataLoader(k3_dataset, batch_size=4, shuffle=False)

    for pyg_b, k3_b in zip(pyg_dl, k3_dl):
        assert pyg_b.num_nodes == k3_b.num_nodes
        assert pyg_b.num_edges == k3_b.num_edges
        assert pyg_b.num_graphs == k3_b.num_graphs
        assert np.array_equal(np.asarray(pyg_b.batch), np.asarray(k3_b.batch))
        assert np.array_equal(np.asarray(pyg_b.ptr), np.asarray(k3_b.ptr))
        assert np.array_equal(np.asarray(pyg_b.edge_index), np.asarray(k3_b.edge_index))
        assert np.allclose(np.asarray(pyg_b.x), np.asarray(k3_b.x))


def test_reference_data_list_loader():
    pyg_dataset = [create_pyg_data(num_nodes=4) for _ in range(4)]
    k3_dataset = [create_k3_data(num_nodes=4) for _ in range(4)]

    pyg_dl = pyg_loader.DataListLoader(pyg_dataset, batch_size=2, shuffle=False)
    k3_dl = k3_loader.DataListLoader(k3_dataset, batch_size=2, shuffle=False)

    for pyg_b, k3_b in zip(pyg_dl, k3_dl):
        assert len(pyg_b) == len(k3_b) == 2


def test_reference_dense_data_loader():
    def make_pyg_dense():
        return pyg_data.Data(x=torch.randn(4, 6), adj=torch.randn(4, 4))

    def make_k3_dense():
        return k3_data.Data(x=torch.randn(4, 6), adj=torch.randn(4, 4))

    torch.manual_seed(123)
    pyg_ds = [make_pyg_dense() for _ in range(4)]
    torch.manual_seed(123)
    k3_ds = [make_k3_dense() for _ in range(4)]

    pyg_dl = pyg_loader.DenseDataLoader(pyg_ds, batch_size=2, shuffle=False)
    k3_dl = k3_loader.DenseDataLoader(k3_ds, batch_size=2, shuffle=False)

    pyg_b = next(iter(pyg_dl))
    k3_b = next(iter(k3_dl))

    assert pyg_b.x.shape == k3_b.x.shape
    assert pyg_b.adj.shape == k3_b.adj.shape


def test_reference_temporal_dataloader():
    src = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    dst = torch.tensor([1, 2, 2, 3], dtype=torch.long)
    t = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    msg = torch.randn(4, 4)

    pyg_t = pyg_data.TemporalData(src=src, dst=dst, t=t, msg=msg)
    k3_t = k3_data.TemporalData(src=src, dst=dst, t=t, msg=msg)

    pyg_dl = pyg_loader.TemporalDataLoader(pyg_t, batch_size=2, neg_sampling_ratio=0.0)
    k3_dl = k3_loader.TemporalDataLoader(k3_t, batch_size=2, neg_sampling_ratio=0.0)

    for pyg_b, k3_b in zip(pyg_dl, k3_dl):
        assert torch.equal(pyg_b.src, k3_b.src)
        assert torch.equal(pyg_b.dst, k3_b.dst)
        assert torch.equal(pyg_b.n_id, k3_b.n_id)


def test_reference_random_node_loader():
    torch.manual_seed(42)
    d = create_k3_data(num_nodes=10)
    loader = k3_loader.RandomNodeLoader(d, num_parts=2)

    batches = list(loader)
    assert len(batches) == 2
    assert sum(b.num_nodes for b in batches) >= 10


def test_reference_dynamic_batch_sampler():
    pyg_ds = [create_pyg_data(num_nodes=i + 2) for i in range(10)]
    k3_ds = [create_k3_data(num_nodes=i + 2) for i in range(10)]

    pyg_sampler = pyg_loader.DynamicBatchSampler(pyg_ds, max_num=12, mode='node', num_steps=3)
    k3_sampler = k3_loader.DynamicBatchSampler(k3_ds, max_num=12, mode='node', num_steps=3)

    pyg_batches = list(pyg_sampler)
    k3_batches = list(k3_sampler)

    assert pyg_batches == k3_batches


def test_reference_imbalanced_sampler():
    dataset = [create_k3_data(num_nodes=4) for _ in range(8)]
    for d in dataset[:6]:
        d.y = torch.tensor([0], dtype=torch.long)
    for d in dataset[6:]:
        d.y = torch.tensor([1], dtype=torch.long)

    sampler = k3_loader.ImbalancedSampler(dataset, num_samples=10)
    assert len(list(sampler)) == 10


def test_reference_zip_loader():
    d = create_k3_data(num_nodes=10)
    l1 = k3_loader.NeighborLoader(d, num_neighbors=[2], input_nodes=torch.arange(4))
    l2 = k3_loader.NeighborLoader(d, num_neighbors=[2], input_nodes=torch.arange(4, 8))

    zl = k3_loader.ZipLoader([l1, l2], batch_size=2)
    batches = list(zl)
    assert len(batches) == 2


def test_benchmark_dataloader():
    num_graphs = 200
    pyg_dataset = [create_pyg_data(num_nodes=20, num_features=16) for _ in range(num_graphs)]
    k3_dataset = [create_k3_data(num_nodes=20, num_features=16) for _ in range(num_graphs)]

    # Benchmark PyG
    pyg_dl = pyg_loader.DataLoader(pyg_dataset, batch_size=16, shuffle=False)
    t0 = time.perf_counter()
    for _ in pyg_dl:
        pass
    pyg_time = time.perf_counter() - t0

    # Benchmark k3_node
    k3_dl = k3_loader.DataLoader(k3_dataset, batch_size=16, shuffle=False)
    t0 = time.perf_counter()
    for _ in k3_dl:
        pass
    k3_time = time.perf_counter() - t0

    print(f"\nDataLoader Benchmark (200 graphs, batch_size=16):")
    print(f"PyG DataLoader:    {pyg_time * 1000:.2f} ms")
    print(f"k3_node DataLoader: {k3_time * 1000:.2f} ms")
    assert k3_time < 5.0  # Reasonable throughput threshold
