import numpy as np
import pytest
import torch
import torch_geometric.data as pyg_data
from keras import ops

import k3_node.data as k3_data


def test_reference_data_parity():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
    y = torch.tensor([0, 1, 0], dtype=torch.int64)

    pyg_d = pyg_data.Data(x=x, edge_index=edge_index, y=y)
    k3_d = k3_data.Data(x=x, edge_index=edge_index, y=y)

    assert pyg_d.num_nodes == k3_d.num_nodes == 3
    assert pyg_d.num_edges == k3_d.num_edges == 4
    assert pyg_d.num_features == k3_d.num_features == 2
    assert k3_d.num_classes == 2
    assert pyg_d.is_undirected() == k3_d.is_undirected() is True
    assert pyg_d.has_self_loops() == k3_d.has_self_loops() is False
    assert pyg_d.has_isolated_nodes() == k3_d.has_isolated_nodes() is False

    # Check clone and dict
    assert set(pyg_d.keys()) == set(k3_d.keys())
    assert np.allclose(pyg_d.x.numpy(), ops.convert_to_numpy(k3_d.x))
    assert np.allclose(pyg_d.edge_index.numpy(), ops.convert_to_numpy(k3_d.edge_index))


def test_reference_batch_parity():
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    ei1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    x2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)
    ei2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)

    pyg_d1 = pyg_data.Data(x=x1, edge_index=ei1)
    pyg_d2 = pyg_data.Data(x=x2, edge_index=ei2)
    pyg_batch = pyg_data.Batch.from_data_list([pyg_d1, pyg_d2])

    k3_d1 = k3_data.Data(x=x1, edge_index=ei1)
    k3_d2 = k3_data.Data(x=x2, edge_index=ei2)
    k3_batch = k3_data.Batch.from_data_list([k3_d1, k3_d2])

    assert pyg_batch.num_graphs == k3_batch.num_graphs == 2
    assert pyg_batch.num_nodes == k3_batch.num_nodes == 5
    assert pyg_batch.num_edges == k3_batch.num_edges == 5

    assert np.allclose(pyg_batch.batch.numpy(), ops.convert_to_numpy(k3_batch.batch))
    assert np.allclose(pyg_batch.ptr.numpy(), ops.convert_to_numpy(k3_batch.ptr))
    assert np.allclose(pyg_batch.x.numpy(), ops.convert_to_numpy(k3_batch.x))
    assert np.allclose(pyg_batch.edge_index.numpy(), ops.convert_to_numpy(k3_batch.edge_index))

    # Separation parity
    pyg_rec1 = pyg_batch[0]
    k3_rec1 = k3_batch[0]
    assert np.allclose(pyg_rec1.x.numpy(), ops.convert_to_numpy(k3_rec1.x))
    assert np.allclose(pyg_rec1.edge_index.numpy(), ops.convert_to_numpy(k3_rec1.edge_index))


def test_reference_hetero_data_parity():
    pyg_h = pyg_data.HeteroData()
    pyg_h["paper"].x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    pyg_h["author"].x = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)
    pyg_h["author", "writes", "paper"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.int64)

    k3_h = k3_data.HeteroData()
    k3_h["paper"].x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    k3_h["author"].x = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)
    k3_h["author", "writes", "paper"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 1]], dtype=torch.int64)

    assert set(pyg_h.node_types) == set(k3_h.node_types)
    assert set(pyg_h.edge_types) == set(k3_h.edge_types)
    assert pyg_h.metadata() == k3_h.metadata()

    # Homogeneous conversion parity
    pyg_homo = pyg_h.to_homogeneous()
    k3_homo = k3_h.to_homogeneous()
    assert pyg_homo.num_nodes == k3_homo.num_nodes == 5
    assert pyg_homo.num_edges == k3_homo.num_edges == 3
    assert np.allclose(pyg_homo.edge_index.numpy(), ops.convert_to_numpy(k3_homo.edge_index))
