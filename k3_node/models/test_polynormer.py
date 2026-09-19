import pytest
from keras import ops

from k3_node.models import Polynormer


@pytest.mark.parametrize('local_attn', [True, False])
@pytest.mark.parametrize('qk_shared', [True, False])
@pytest.mark.parametrize('pre_ln', [True, False])
@pytest.mark.parametrize('post_bn', [True, False])
def test_polynormer_local(local_attn, qk_shared, pre_ln, post_bn):
    """Local-mode Polynormer (default _global=False) should output [N, out_channels]."""
    x = ops.ones((10, 16))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])
    batch = ops.convert_to_tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = Polynormer(
        in_channels=16,
        hidden_channels=32,
        out_channels=40,
        local_layers=2,
        global_layers=1,
        qk_shared=qk_shared,
        pre_ln=pre_ln,
        post_bn=post_bn,
        local_attn=local_attn,
        heads=2,
    )
    out = model(x, edge_index, batch)
    assert ops.shape(out) == (10, 40)


@pytest.mark.parametrize('local_attn', [True, False])
@pytest.mark.parametrize('qk_shared', [True, False])
def test_polynormer_global(local_attn, qk_shared):
    """Global-mode Polynormer (_global=True) should output [N, out_channels]."""
    x = ops.ones((10, 16))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])
    batch = ops.convert_to_tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = Polynormer(
        in_channels=16,
        hidden_channels=32,
        out_channels=40,
        local_layers=2,
        global_layers=1,
        qk_shared=qk_shared,
        local_attn=local_attn,
    )
    model._global = True
    out = model(x, edge_index, batch)
    assert ops.shape(out) == (10, 40)


def test_polynormer_log_softmax_sums():
    """Output of Polynormer is log_softmax, so exp(out).sum(axis=-1) ≈ 1."""
    import numpy as np
    x = ops.ones((6, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]])
    batch = ops.convert_to_tensor([0, 0, 0, 0, 0, 0])

    model = Polynormer(
        in_channels=8,
        hidden_channels=16,
        out_channels=5,
        local_layers=1,
        global_layers=1,
    )
    out = model(x, edge_index, batch)
    assert ops.shape(out) == (6, 5)

    out_np = ops.convert_to_numpy(out)
    sums = np.exp(out_np).sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_polynormer_global_toggle():
    """Switching between local and global modes on the same model instance."""
    x = ops.ones((8, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    batch = ops.convert_to_tensor([0, 0, 0, 0, 1, 1, 1, 1])

    model = Polynormer(
        in_channels=16,
        hidden_channels=16,
        out_channels=10,
        local_layers=1,
        global_layers=1,
    )

    model._global = False
    out_local = model(x, edge_index, batch)
    assert ops.shape(out_local) == (8, 10)

    model._global = True
    out_global = model(x, edge_index, batch)
    assert ops.shape(out_global) == (8, 10)

