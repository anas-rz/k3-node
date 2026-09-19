from keras import ops, random
from k3_node.models import NeuralFingerprint


def test_neural_fingerprint():
    model = NeuralFingerprint(in_channels=16, hidden_channels=32, out_channels=8, num_layers=3)
    x = random.normal((6, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype="int64")
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")

    out = model(x, edge_index, batch)
    assert out.shape == (2, 8)

