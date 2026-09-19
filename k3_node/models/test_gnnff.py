import keras.ops as ops
from k3_node.models.gnnff import GNNFF, GaussianFilter


def test_gaussian_filter():
    gf = GaussianFilter(start=0.0, stop=5.0, num_gaussians=10)
    dist = ops.convert_to_tensor([0.5, 1.5, 3.0])
    out = gf(dist)
    assert out.shape == (3, 10)


def test_gnnff():
    z = ops.convert_to_tensor([1, 6, 8, 1])
    pos = ops.convert_to_tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype="float32")

    model = GNNFF(hidden_node_channels=16, hidden_edge_channels=16, num_layers=2)
    force = model(z, pos)
    assert force.shape == (4, 3)

