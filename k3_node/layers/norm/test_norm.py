import numpy as np
from keras import ops, random

from k3_node.layers import MessageNorm, MeanSubtractionNorm


def test_message_norm():
    norm = MessageNorm(learn_scale=True)
    x = random.normal((100, 16))
    msg = random.normal((100, 16))
    out = norm((x, msg))
    assert ops.shape(out) == (100, 16)

    norm = MessageNorm(learn_scale=False)
    out = norm((x, msg))
    assert ops.shape(out) == (100, 16)


def test_mean_subtraction_norm():
    x = random.normal((6, 16))

    norm = MeanSubtractionNorm()

    out = norm(x)
    assert ops.shape(out) == (6, 16)
    assert np.allclose(ops.mean(out), ops.convert_to_numpy(0.0), atol=1e-6)
