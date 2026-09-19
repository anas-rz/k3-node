import numpy as np
import pytest
from keras import ops

from k3_node.data import Batch, Data, HeteroData


def test_batch_homogeneous():
    d1 = Data(
        x=ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=ops.convert_to_tensor([[0, 1], [1, 0]], dtype="int64"),
        y=ops.convert_to_tensor([0]),
    )
    d2 = Data(
        x=ops.convert_to_tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
        edge_index=ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype="int64"),
        y=ops.convert_to_tensor([1]),
    )

    batch = Batch.from_data_list([d1, d2])
    assert batch.num_graphs == 2
    assert batch.num_nodes == 5
    assert batch.num_edges == 5

    # Check batch vector
    batch_vec = ops.convert_to_numpy(batch.batch)
    assert np.array_equal(batch_vec, [0, 0, 1, 1, 1])

    # Check ptr vector
    ptr_vec = ops.convert_to_numpy(batch.ptr)
    assert np.array_equal(ptr_vec, [0, 2, 5])

    # Check offset edge_index
    ei = ops.convert_to_numpy(batch.edge_index)
    assert np.array_equal(ei[:, :2], [[0, 1], [1, 0]])
    assert np.array_equal(ei[:, 2:], [[2, 3, 4], [3, 4, 2]])

    # Separate back
    rec1 = batch[0]
    rec2 = batch[1]
    assert rec1.num_nodes == 2
    assert rec2.num_nodes == 3
    assert np.allclose(ops.convert_to_numpy(rec1.x), ops.convert_to_numpy(d1.x))
    assert np.allclose(ops.convert_to_numpy(rec2.x), ops.convert_to_numpy(d2.x))

    data_list = batch.to_data_list()
    assert len(data_list) == 2


def test_batch_heterogeneous():
    h1 = HeteroData()
    h1["v"].x = ops.convert_to_tensor([[1.0], [2.0]])
    h1["v", "e", "v"].edge_index = ops.convert_to_tensor([[0], [1]], dtype="int64")

    h2 = HeteroData()
    h2["v"].x = ops.convert_to_tensor([[3.0], [4.0], [5.0]])
    h2["v", "e", "v"].edge_index = ops.convert_to_tensor([[0, 1], [1, 2]], dtype="int64")

    batch = Batch.from_data_list([h1, h2])
    assert batch.num_graphs == 2
    assert batch["v"].num_nodes == 5
    assert batch["v", "e", "v"].num_edges == 3

    rec1 = batch[0]
    assert rec1["v"].num_nodes == 2
    assert rec1["v", "e", "v"].num_edges == 1

