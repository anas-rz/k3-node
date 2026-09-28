import numpy as np
from keras import ops

from k3_node.models.lpformer import LPFormer, get_ppr


def test_get_ppr():
    ppr = get_ppr(np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]), num_nodes=3)
    assert ppr.shape == (3, 3)
    np.testing.assert_allclose(np.asarray(ppr.sum(axis=1)).ravel(), 1.0, atol=0.05)  # (approximate) distributions


def test_lpformer():
    rng = np.random.default_rng(0)
    x = rng.random((20, 16)).astype("float32")
    edge_index = rng.integers(0, 20, (2, 60))
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    batch = np.array([[0, 1, 5], [2, 3, 9]])
    model = LPFormer(in_channels=16, hidden_channels=16, num_gnn_layers=2)
    ppr = model.calc_sparse_ppr(edge_index, 20)
    assert tuple(model(batch, x, edge_index, ppr_matrix=ppr).shape) == (3,)
    assert tuple(model(batch, x, edge_index, ppr_matrix=ppr, training=True).shape) == (3,)
