import keras.ops as ops
from k3_node.models.lpformer import LPFormer, MLP, compute_ppr_matrix


def test_compute_ppr_matrix():
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    ppr = compute_ppr_matrix(edge_index, num_nodes=3)
    assert ppr.shape == (3, 3)


def test_lpformer():
    x = ops.ones((4, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    batch = ops.convert_to_tensor([[0, 1], [2, 3]])

    model = LPFormer(in_channels=16, hidden_channels=16, num_gnn_layers=2)
    out = model(batch, x, edge_index)
    assert out.shape == (2, 1)

