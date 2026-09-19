import keras.ops as ops
from k3_node.models.visnet import ViSNet, CosineCutoff, Sphere


def test_cosine_cutoff():
    cutoff = CosineCutoff(5.0)
    dist = ops.convert_to_tensor([0.0, 2.5, 5.0, 6.0])
    out = cutoff(dist)
    assert float(out[0]) == 1.0
    assert float(out[2]) == 0.0
    assert float(out[3]) == 0.0


def test_sphere():
    sphere1 = Sphere(lmax=1)
    v = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    sh1 = sphere1(v)
    assert sh1.shape == (1, 3)

    sphere2 = Sphere(lmax=2)
    sh2 = sphere2(v)
    assert sh2.shape == (1, 8)


def test_visnet():
    z = ops.convert_to_tensor([1, 6, 8, 1])
    pos = ops.convert_to_tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype="float32")

    model = ViSNet(
        lmax=1,
        num_heads=2,
        num_layers=2,
        hidden_channels=16,
        num_rbf=8,
        cutoff=5.0,
    )
    y, dy = model(z, pos)
    assert y.shape == (1, 1)

