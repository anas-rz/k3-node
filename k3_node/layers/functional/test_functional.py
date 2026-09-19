from keras import ops

from k3_node.layers.functional import bro, gini


def test_bro():
    batch = ops.convert_to_tensor([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype="int64")

    g1 = ops.convert_to_tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
        [0.2, 0.2, 0.0, 0.2],
    ], dtype="float32")
    g2 = ops.convert_to_tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.0, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ], dtype="float32")
    g3 = ops.convert_to_tensor([
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2],
    ], dtype="float32")
    x = ops.concatenate([g1, g2, g3], axis=0)

    out = bro(x, batch)
    assert ops.shape(out) == ()
    assert float(out) > 0.0


def test_gini():
    w = ops.convert_to_tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1000.0]], dtype="float32")
    out = gini(w)
    assert abs(float(out) - 0.5) < 1e-5
