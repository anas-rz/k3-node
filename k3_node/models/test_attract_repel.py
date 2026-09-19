from keras import ops

from k3_node.models import ARLinkPredictor


def test_ar_link_predictor():
    model = ARLinkPredictor(in_channels=16, hidden_channels=32, num_layers=2)
    x = ops.convert_to_tensor([[0.1] * 16] * 4, dtype="float32")
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]], dtype="int64")

    pred = model(x, edge_index)
    assert ops.shape(pred)[0] == ops.shape(edge_index)[1]
    pred_np = ops.convert_to_numpy(pred)
    assert (pred_np >= 0).all() and (pred_np <= 1).all()

    attract_z, repel_z = model.encode(x)
    assert ops.shape(attract_z) == (4, 16)
    assert ops.shape(repel_z) == (4, 16)

    raw_scores = model.decode(attract_z, repel_z, edge_index)
    assert ops.shape(raw_scores)[0] == ops.shape(edge_index)[1]

    r_fraction = model.calculate_r_fraction(attract_z, repel_z)
    assert 0 <= r_fraction <= 1


def test_ar_link_predictor_with_custom_ratio():
    model = ARLinkPredictor(in_channels=8, hidden_channels=20, attract_ratio=0.7)
    x = ops.convert_to_tensor([[0.1] * 8] * 5, dtype="float32")

    attract_z, repel_z = model.encode(x)
    assert ops.shape(attract_z) == (5, 14)
    assert ops.shape(repel_z) == (5, 6)
