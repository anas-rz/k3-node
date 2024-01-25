from keras import ops, random

from k3_node.layers import MessageNorm


def test_message_norm():
    norm = MessageNorm(learn_scale=True)
    x = random.normal((100, 16))
    msg = random.normal((100, 16))
    out = norm((x, msg))
    assert ops.shape(out) == (100, 16)

    norm = MessageNorm(learn_scale=False)
    out = norm((x, msg))
    assert ops.shape(out) == (100, 16)
