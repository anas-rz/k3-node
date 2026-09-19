from keras import ops

from k3_node.models import AttentiveFP


def test_attentive_fp():
    model = AttentiveFP(8, 16, 32, edge_dim=3, num_layers=2, num_timesteps=2)
    assert str(model) == (
        "AttentiveFP(in_channels=8, hidden_channels=16, "
        "out_channels=32, edge_dim=3, num_layers=2, "
        "num_timesteps=2)"
    )

    x = ops.convert_to_tensor([[1.0] * 8] * 4, dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]], dtype="int64")
    edge_attr = ops.convert_to_tensor([[1.0] * 3] * 6, dtype="float32")
    batch = ops.convert_to_tensor([0, 0, 0, 0], dtype="int64")

    out = model(x, edge_index, edge_attr, batch)
    assert ops.shape(out) == (1, 32)


def test_attentive_fp_multi_graph_batch():
    model = AttentiveFP(4, 8, 2, edge_dim=2, num_layers=2, num_timesteps=2)

    x = ops.convert_to_tensor([[1.0] * 4] * 6, dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 1, 3, 4], [1, 0, 4, 3]], dtype="int64")
    edge_attr = ops.convert_to_tensor([[1.0] * 2] * 4, dtype="float32")
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")

    out = model(x, edge_index, edge_attr, batch)
    assert ops.shape(out) == (2, 2)
