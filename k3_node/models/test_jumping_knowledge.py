import pytest
from keras import ops

from k3_node.models import JumpingKnowledge, HeteroJumpingKnowledge


def test_jumping_knowledge_cat():
    num_nodes, channels, num_layers = 100, 17, 5
    xs = [ops.ones((num_nodes, channels)) for _ in range(num_layers)]

    model = JumpingKnowledge('cat')
    assert str(model) == 'JumpingKnowledge(cat)'

    out = model(xs)
    assert ops.shape(out) == (num_nodes, channels * num_layers)


def test_jumping_knowledge_max():
    num_nodes, channels, num_layers = 100, 17, 5
    xs = [ops.ones((num_nodes, channels)) for _ in range(num_layers)]

    model = JumpingKnowledge('max')
    assert str(model) == 'JumpingKnowledge(max)'

    out = model(xs)
    assert ops.shape(out) == (num_nodes, channels)


def test_jumping_knowledge_lstm():
    num_nodes, channels, num_layers = 100, 17, 5
    xs = [ops.ones((num_nodes, channels)) for _ in range(num_layers)]

    model = JumpingKnowledge('lstm', channels, num_layers)
    assert str(model) == (
        f'JumpingKnowledge(lstm, channels={channels}, layers={num_layers})'
    )

    out = model(xs)
    assert ops.shape(out) == (num_nodes, channels)


def test_jumping_knowledge_invalid_mode():
    with pytest.raises(AssertionError):
        JumpingKnowledge('invalid')


def test_jumping_knowledge_lstm_missing_args():
    with pytest.raises(AssertionError):
        JumpingKnowledge('lstm')


def test_hetero_jumping_knowledge_cat():
    num_nodes, channels, num_layers = 100, 17, 5
    types = ["author", "paper"]
    xs_dict = {
        key: [ops.ones((num_nodes, channels)) for _ in range(num_layers)]
        for key in types
    }

    model = HeteroJumpingKnowledge(types, mode='cat')
    model.reset_parameters()
    assert str(model) == 'HeteroJumpingKnowledge(num_types=2, mode=cat)'

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert ops.shape(out) == (num_nodes, channels * num_layers)


def test_hetero_jumping_knowledge_max():
    num_nodes, channels, num_layers = 100, 17, 5
    types = ["author", "paper"]
    xs_dict = {
        key: [ops.ones((num_nodes, channels)) for _ in range(num_layers)]
        for key in types
    }

    model = HeteroJumpingKnowledge(types, mode='max')
    assert str(model) == 'HeteroJumpingKnowledge(num_types=2, mode=max)'

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert ops.shape(out) == (num_nodes, channels)


def test_hetero_jumping_knowledge_lstm():
    num_nodes, channels, num_layers = 50, 8, 3
    types = ["author", "paper"]
    xs_dict = {
        key: [ops.ones((num_nodes, channels)) for _ in range(num_layers)]
        for key in types
    }

    model = HeteroJumpingKnowledge(types, mode='lstm', channels=channels,
                                   num_layers=num_layers)
    assert str(model) == (
        f'HeteroJumpingKnowledge(num_types=2, mode=lstm, '
        f'channels={channels}, layers={num_layers})'
    )

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert ops.shape(out) == (num_nodes, channels)


def test_jumping_knowledge_cat_aggregates_correctly():
    """Values in cat mode should be the concatenation of all layer reps."""
    xs = [
        ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]]),
        ops.convert_to_tensor([[5.0, 6.0], [7.0, 8.0]]),
    ]
    model = JumpingKnowledge('cat')
    out = model(xs)
    assert ops.shape(out) == (2, 4)


def test_jumping_knowledge_max_aggregates_correctly():
    """Values in max mode should be the element-wise max across layers."""
    xs = [
        ops.convert_to_tensor([[1.0, 8.0], [3.0, 4.0]]),
        ops.convert_to_tensor([[5.0, 6.0], [7.0, 2.0]]),
    ]
    import numpy as np
    model = JumpingKnowledge('max')
    out = model(xs)
    assert ops.shape(out) == (2, 2)
    out_np = ops.convert_to_numpy(out)
    assert np.allclose(out_np[0], [5.0, 8.0])
    assert np.allclose(out_np[1], [7.0, 4.0])

