import numpy as np
import pytest
from keras import ops

from k3_node.data import Data


def test_data_basic():
    x = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype="int64")
    y = ops.convert_to_tensor([0, 1, 0], dtype="int64")

    data = Data(x=x, edge_index=edge_index, y=y)
    assert data.num_nodes == 3
    assert data.num_edges == 4
    assert data.num_features == 2
    assert data.num_classes == 2
    assert data.is_undirected()
    assert not data.is_directed()
    assert not data.has_self_loops()
    assert not data.has_isolated_nodes()

    # Attribute and dict access
    assert "x" in data
    assert "edge_index" in data
    assert "edge_attr" not in data
    assert len(data.keys()) == 3
    assert ops.convert_to_numpy(data["x"]).shape == (3, 2)
    assert ops.convert_to_numpy(data.x).shape == (3, 2)

    # Clone
    clone = data.clone()
    assert clone.num_nodes == 3
    assert clone.num_edges == 4

    # Dict and namedtuple
    d = data.to_dict()
    assert "x" in d and "edge_index" in d and "y" in d
    nt = data.to_namedtuple()
    assert hasattr(nt, "x") and hasattr(nt, "edge_index")

    # String repr
    repr_str = str(data)
    assert "Data(" in repr_str and "x=" in repr_str and "edge_index=" in repr_str


def test_data_subgraph():
    x = ops.convert_to_tensor([[10.0], [20.0], [30.0], [40.0]])
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
    data = Data(x=x, edge_index=edge_index)

    subset = ops.convert_to_tensor([0, 2], dtype="int64")
    sub = data.subgraph(subset)
    assert sub.num_nodes == 2
    assert ops.convert_to_numpy(sub.x).tolist() == [[10.0], [30.0]]


def test_data_to_heterogeneous():
    x = ops.convert_to_tensor([[1.0], [2.0]])
    edge_index = ops.convert_to_tensor([[0, 1], [1, 0]], dtype="int64")
    data = Data(x=x, edge_index=edge_index)

    hetero = data.to_heterogeneous(node_type="v", edge_type=("v", "e", "v"))
    assert "v" in hetero.node_types
    assert ("v", "e", "v") in hetero.edge_types
    assert ops.convert_to_numpy(hetero["v"].x).shape == (2, 1)
    assert ops.convert_to_numpy(hetero["v", "e", "v"].edge_index).shape == (2, 2)

