import keras.ops as ops
from k3_node.models.schnet import SchNet, ShiftedSoftplus, GaussianSmearing


def test_shifted_softplus():
    act = ShiftedSoftplus()
    x = ops.convert_to_tensor([0.0, 1.0, 2.0])
    out = act(x)
    assert out.shape == (3,)
    assert float(out[0]) == 0.0


def test_gaussian_smearing():
    smear = GaussianSmearing(start=0.0, stop=5.0, num_gaussians=10)
    dist = ops.convert_to_tensor([0.0, 1.0, 2.5])
    out = smear(dist)
    assert out.shape == (3, 10)


def test_schnet():
    z = ops.convert_to_tensor([1, 6, 8, 1])
    pos = ops.convert_to_tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype="float32")

    model = SchNet(
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=10,
        cutoff=5.0,
    )
    out = model(z, pos)
    assert out.shape == (1, 1)

    # Test with batch vector
    batch = ops.convert_to_tensor([0, 0, 1, 1])
    out_batched = model(z, pos, batch=batch)
    assert out_batched.shape == (2, 1)

