import numpy as np
from keras import ops, random


from k3_node.layers import MeanSubtractionNorm


def test_mean_subtraction_norm():
    x = random.normal((6, 16))

    norm = MeanSubtractionNorm()

    out = norm(x)
    assert ops.shape(out) == (6, 16)
    assert np.allclose(ops.mean(out), ops.convert_to_numpy(0.), atol=1e-6)