import pytest
from keras import ops

from k3_node.models import PMLP


def test_pmlp():
    x = ops.ones((4, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    pmlp = PMLP(in_channels=16, hidden_channels=32, out_channels=2, num_layers=4)
    assert str(pmlp) == 'PMLP(16, 2, num_layers=4)'

    pmlp.training = True
    out = pmlp(x)
    assert ops.shape(out) == (4, 2)

    pmlp.training = False
    out = pmlp(x, edge_index)
    assert ops.shape(out) == (4, 2)


def test_pmlp_raises_without_edge_index():
    """Should raise ValueError when edge_index is missing during inference."""
    x = ops.ones((4, 16))
    pmlp = PMLP(in_channels=16, hidden_channels=32, out_channels=2, num_layers=4)

    with pytest.raises(ValueError, match="'edge_index' needs to be present"):
        pmlp.training = False
        pmlp(x)


def test_pmlp_call_training_override():
    """The training kwarg at call-time should override self.training."""
    x = ops.ones((4, 16))
    edge_index = ops.convert_to_tensor([[0, 1], [1, 0]])
    pmlp = PMLP(in_channels=16, hidden_channels=32, out_channels=4, num_layers=2)

    # Instance says training=True but call says training=False (requires edge_index)
    pmlp.training = True
    out = pmlp(x, edge_index, training=False)
    assert ops.shape(out) == (4, 4)

    # Instance says training=False but call says training=True (no edge_index needed)
    pmlp.training = False
    out = pmlp(x, training=True)
    assert ops.shape(out) == (4, 4)


@pytest.mark.parametrize("norm", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_pmlp_configs(norm, bias, dropout):
    x = ops.ones((8, 10))
    pmlp = PMLP(
        in_channels=10,
        hidden_channels=20,
        out_channels=5,
        num_layers=3,
        norm=norm,
        bias=bias,
        dropout=dropout,
    )
    pmlp.training = True
    out = pmlp(x)
    assert ops.shape(out) == (8, 5)


def test_pmlp_single_layer():
    """Edge case: num_layers=1 should work (no hidden layers, direct in→out)."""
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1], [1, 0]])
    pmlp = PMLP(in_channels=8, hidden_channels=16, out_channels=3, num_layers=1)
    pmlp.training = True
    out = pmlp(x)
    assert ops.shape(out) == (4, 3)

    pmlp.training = False
    out = pmlp(x, edge_index)
    assert ops.shape(out) == (4, 3)

