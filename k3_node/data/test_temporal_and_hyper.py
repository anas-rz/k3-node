import numpy as np
import pytest
from keras import ops

from k3_node.data import HypergraphData, TemporalData


def test_temporal_data():
    src = ops.convert_to_tensor([0, 1, 2, 3], dtype="int64")
    dst = ops.convert_to_tensor([1, 2, 3, 4], dtype="int64")
    t = ops.convert_to_tensor([10, 20, 30, 40], dtype="int64")
    msg = ops.convert_to_tensor([[1.0], [2.0], [3.0], [4.0]])

    data = TemporalData(src=src, dst=dst, t=t, msg=msg)
    assert data.num_events == 4
    assert data.num_nodes == 5
    assert len(data) == 4

    sub = data[:2]
    assert sub.num_events == 2
    assert ops.convert_to_numpy(sub.src).tolist() == [0, 1]


def test_hypergraph_data():
    x = ops.convert_to_tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    edge_index = ops.convert_to_tensor(
        [[0, 1, 2, 1, 2, 3, 4], [0, 0, 0, 1, 1, 1, 1]], dtype="int64"
    )
    data = HypergraphData(x=x, edge_index=edge_index)
    assert data.num_nodes == 5
    assert data.num_edges == 2

