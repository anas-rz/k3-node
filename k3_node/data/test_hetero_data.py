import numpy as np
import pytest
from keras import ops

from k3_node.data import HeteroData


def test_hetero_data_basic():
    data = HeteroData()

    data["paper"].x = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    data["author"].x = ops.convert_to_tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    data["author", "writes", "paper"].edge_index = ops.convert_to_tensor(
        [[0, 1, 2], [0, 1, 1]], dtype="int64"
    )

    assert set(data.node_types) == {"paper", "author"}
    assert set(data.edge_types) == {("author", "writes", "paper")}
    assert data["paper"].num_nodes == 2
    assert data["author"].num_nodes == 3
    assert data["author", "writes", "paper"].num_edges == 3

    meta = data.metadata()
    assert len(meta[0]) == 2
    assert len(meta[1]) == 1

    # Homogeneous conversion
    homo = data.to_homogeneous()
    assert homo.num_nodes == 5
    assert homo.num_edges == 3
    assert ops.convert_to_numpy(homo.x).shape == (5, 2)
    assert ops.convert_to_numpy(homo.edge_index).shape == (2, 3)

