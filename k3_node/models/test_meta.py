import pytest
import numpy as np
import keras
from keras import ops

from k3_node.models import MetaLayer
from k3_node.layers.conv.utils import scatter


# Global counter to verify model call counts
_count = 0


def test_meta_layer_repr():
    """MetaLayer should have the correct string representation."""
    assert str(MetaLayer()) == (
        'MetaLayer(\n'
        '  edge_model=None,\n'
        '  node_model=None,\n'
        '  global_model=None\n'
        ')'
    )


def test_meta_layer_none_models():
    """All None models should return unchanged tensors."""
    x = ops.ones((20, 10))
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]])

    model = MetaLayer()
    x_out, edge_attr_out, u_out = model(x, edge_index)

    assert ops.shape(x_out) == (20, 10)
    assert edge_attr_out is None
    assert u_out is None


def test_meta_layer_call_counting():
    """Verify that the correct submodels are called for each combination."""
    global _count
    _count = 0

    def dummy_model(*args):
        global _count
        _count += 1
        return None

    x = ops.ones((20, 10))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    for edge_model in (dummy_model, None):
        for node_model in (dummy_model, None):
            for global_model in (dummy_model, None):
                model = MetaLayer(edge_model, node_model, global_model)
                out = model(x, edge_index)
                assert isinstance(out, tuple) and len(out) == 3

    assert _count == 12  # 3 models × 4 combinations with at least 1 non-None


def test_meta_layer_with_edge_model():
    """Edge model should receive correct src/dst/edge_attr/u/batch args."""
    received = {}

    def edge_model(src, dst, edge_attr, u, batch):
        received['src_shape'] = ops.shape(src)
        received['dst_shape'] = ops.shape(dst)
        received['edge_attr_shape'] = ops.shape(edge_attr) if edge_attr is not None else None
        return edge_attr  # pass-through

    x = ops.ones((5, 4))
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = ops.zeros((3, 7))

    model = MetaLayer(edge_model=edge_model)
    model(x, edge_index, edge_attr=edge_attr)

    assert received['src_shape'] == (3, 4)
    assert received['dst_shape'] == (3, 4)
    assert received['edge_attr_shape'] == (3, 7)


def test_meta_layer_full_example():
    """Full graph network test matching PyG's test_meta_layer_example."""

    class EdgeModel(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.mlp = keras.Sequential([
                keras.layers.Dense(5),
                keras.layers.ReLU(),
                keras.layers.Dense(5),
            ])

        def call(self, src, dst, edge_attr, u, batch):
            assert edge_attr is not None
            assert u is not None
            assert batch is not None
            out = ops.concatenate([src, dst, edge_attr, ops.take(u, batch, axis=0)], axis=1)
            return self.mlp(out)

    class NodeModel(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.mlp1 = keras.Sequential([
                keras.layers.Dense(10),
                keras.layers.ReLU(),
                keras.layers.Dense(10),
            ])
            self.mlp2 = keras.Sequential([
                keras.layers.Dense(10),
                keras.layers.ReLU(),
                keras.layers.Dense(10),
            ])

        def call(self, x, edge_index, edge_attr, u, batch):
            assert edge_attr is not None
            assert u is not None
            assert batch is not None
            row = edge_index[0]
            col = edge_index[1]
            out = ops.concatenate([ops.take(x, row, axis=0), edge_attr], axis=1)
            out = self.mlp1(out)
            out = scatter(out, col, dim_size=ops.shape(x)[0], reduce="mean")
            out = ops.concatenate([x, out, ops.take(u, batch, axis=0)], axis=1)
            return self.mlp2(out)

    class GlobalModel(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.mlp = keras.Sequential([
                keras.layers.Dense(20),
                keras.layers.ReLU(),
                keras.layers.Dense(20),
            ])

        def call(self, x, edge_index, edge_attr, u, batch):
            assert u is not None
            assert batch is not None
            batch_size = ops.shape(u)[0]
            x_mean = scatter(x, batch, dim_size=batch_size, reduce="mean")
            out = ops.concatenate([u, x_mean], axis=1)
            return self.mlp(out)

    op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())

    x = ops.ones((20, 10))
    edge_attr = ops.ones((40, 5))
    u = ops.ones((2, 20))
    batch = ops.convert_to_tensor([0] * 10 + [1] * 10)
    row_idx = list(range(20)) + list(range(20))
    col_idx = list(range(1, 20)) + [0] + list(range(1, 20)) + [0]
    edge_index = ops.convert_to_tensor([row_idx, col_idx])

    x_out, edge_attr_out, u_out = op(x, edge_index, edge_attr, u, batch)
    assert ops.shape(x_out) == (20, 10)
    assert ops.shape(edge_attr_out) == (40, 5)
    assert ops.shape(u_out) == (2, 20)

