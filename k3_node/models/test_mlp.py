import pytest
from keras import ops

from k3_node.models import MLP


@pytest.mark.parametrize("norm", ["batch_norm", None])
@pytest.mark.parametrize("act_first", [False, True])
@pytest.mark.parametrize("plain_last", [False, True])
def test_mlp(norm, act_first, plain_last):
    x = ops.ones((4, 16))

    mlp = MLP([16, 32, 32, 64], norm=norm, act_first=act_first, plain_last=plain_last)
    assert str(mlp) == "MLP(16, 32, 32, 64)"
    out = mlp(x)
    assert ops.shape(out) == (4, 64)

    mlp2 = MLP(
        16,
        hidden_channels=32,
        out_channels=64,
        num_layers=3,
        norm=norm,
        act_first=act_first,
        plain_last=plain_last,
    )
    assert ops.shape(mlp2(x)) == (4, 64)


@pytest.mark.parametrize("norm", ["BatchNorm", "GraphNorm", "InstanceNorm", "LayerNorm"])
def test_batch(norm):
    x = ops.ones((3, 8))
    batch = ops.convert_to_tensor([0, 0, 1], dtype="int64")

    model = MLP(8, hidden_channels=16, out_channels=32, num_layers=2, norm=norm)
    assert model.supports_norm_batch == (norm != "BatchNorm")

    out = model(x, batch=batch)
    assert ops.shape(out) == (3, 32)


def test_mlp_return_emb():
    x = ops.ones((4, 16))

    mlp = MLP([16, 32, 1])

    out, emb = mlp(x, return_emb=True)
    assert ops.shape(out) == (4, 1)
    assert ops.shape(emb) == (4, 32)

    out, emb = mlp(x, return_emb=False)
    assert ops.shape(out) == (4, 1)
    assert emb is None

    out = mlp(x)
    assert ops.shape(out) == (4, 1)


@pytest.mark.parametrize("plain_last", [False, True])
def test_fine_grained_mlp(plain_last):
    mlp = MLP([16, 32, 32, 64], dropout=[0.1, 0.2, 0.3], bias=[False, True, False], plain_last=plain_last)
    assert ops.shape(mlp(ops.ones((4, 16)))) == (4, 64)
