import numpy as np
import pytest
from keras import ops, random

from k3_node.layers.norm import (
    BatchNorm,
    DiffGroupNorm,
    GraphNorm,
    GraphSizeNorm,
    HeteroBatchNorm,
    HeteroLayerNorm,
    InstanceNorm,
    LayerNorm,
    MeanSubtractionNorm,
    MessageNorm,
    PairNorm,
)


def test_graph_size_norm():
    x = random.normal((100, 16))
    batch = ops.repeat(ops.arange(10, dtype="int32"), 10)

    norm = GraphSizeNorm()

    out1 = norm(x)
    assert ops.shape(out1) == (100, 16)

    out2 = norm(x, batch)
    assert ops.shape(out2) == (100, 16)

    # Tuple input
    out3 = norm((x, batch))
    assert np.allclose(ops.convert_to_numpy(out2), ops.convert_to_numpy(out3), atol=1e-6)


@pytest.mark.parametrize("scale_individually", [False, True])
def test_pair_norm(scale_individually):
    x = random.normal((100, 16))
    batch = ops.zeros((100,), dtype="int32")

    norm = PairNorm(scale_individually=scale_individually)

    out1 = norm(x)
    assert ops.shape(out1) == (100, 16)

    x2 = ops.concatenate([x, x], axis=0)
    batch2 = ops.concatenate([batch, batch + 1], axis=0)
    out2 = norm(x2, batch2)

    assert np.allclose(
        ops.convert_to_numpy(out1),
        ops.convert_to_numpy(out2[:100]),
        atol=1e-5,
    )
    assert np.allclose(
        ops.convert_to_numpy(out1),
        ops.convert_to_numpy(out2[100:]),
        atol=1e-5,
    )


def test_mean_subtraction_norm():
    x = random.normal((6, 16))
    batch = ops.convert_to_tensor(np.array([0, 0, 1, 1, 1, 2], dtype=np.int32))

    norm = MeanSubtractionNorm()

    out1 = norm(x)
    assert ops.shape(out1) == (6, 16)
    assert np.allclose(ops.convert_to_numpy(ops.mean(out1)), 0.0, atol=1e-6)

    out2 = norm(x, batch)
    assert ops.shape(out2) == (6, 16)
    assert np.allclose(ops.convert_to_numpy(ops.mean(out2[0:2], axis=0)), 0.0, atol=1e-6)
    assert np.allclose(ops.convert_to_numpy(ops.mean(out2[2:5], axis=0)), 0.0, atol=1e-6)
    assert np.allclose(ops.convert_to_numpy(ops.mean(out2[5:6], axis=0)), 0.0, atol=1e-6)


@pytest.mark.parametrize("learn_scale", [False, True])
def test_message_norm(learn_scale):
    norm = MessageNorm(learn_scale=learn_scale)
    x = random.normal((100, 16))
    msg = random.normal((100, 16))

    out = norm(x, msg)
    assert ops.shape(out) == (100, 16)

    # Tuple calling
    out_tuple = norm((x, msg))
    assert np.allclose(ops.convert_to_numpy(out), ops.convert_to_numpy(out_tuple), atol=1e-6)

    norm.reset_parameters()
    assert np.allclose(ops.convert_to_numpy(norm.scale), 1.0)


def test_graph_norm():
    x = random.normal((200, 16))
    batch = ops.repeat(ops.arange(4, dtype="int32"), 50)

    norm = GraphNorm(16)

    out = norm(x)
    assert ops.shape(out) == (200, 16)
    assert np.allclose(ops.convert_to_numpy(ops.mean(out, axis=0)), 0.0, atol=1e-5)
    assert np.allclose(ops.convert_to_numpy(ops.std(out, axis=0)), 1.0, atol=1e-3)

    out_b = norm(x, batch)
    assert ops.shape(out_b) == (200, 16)
    assert np.allclose(ops.convert_to_numpy(ops.mean(out_b[:50], axis=0)), 0.0, atol=1e-5)
    assert np.allclose(ops.convert_to_numpy(ops.std(out_b[:50], axis=0)), 1.0, atol=1e-3)

    norm.reset_parameters()
    assert np.allclose(ops.convert_to_numpy(norm.weight), 1.0)
    assert np.allclose(ops.convert_to_numpy(norm.bias), 0.0)
    assert np.allclose(ops.convert_to_numpy(norm.mean_scale), 1.0)


@pytest.mark.parametrize("conf", [True, False])
def test_instance_norm(conf):
    batch = ops.zeros((100,), dtype="int32")
    x1 = random.normal((100, 16))
    x2 = random.normal((100, 16))

    norm1 = InstanceNorm(16, affine=conf, track_running_stats=conf)
    norm2 = InstanceNorm(16, affine=conf, track_running_stats=conf)

    out1 = norm1(x1)
    out2 = norm2(x1, batch)
    assert ops.shape(out1) == (100, 16)
    assert np.allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2), atol=1e-5)

    if conf:
        assert np.allclose(
            ops.convert_to_numpy(norm1.running_mean),
            ops.convert_to_numpy(norm2.running_mean),
            atol=1e-5,
        )
        assert np.allclose(
            ops.convert_to_numpy(norm1.running_var),
            ops.convert_to_numpy(norm2.running_var),
            atol=1e-5,
        )

    out1_eval = norm1(x1, training=False)
    out2_eval = norm2(x1, batch, training=False)
    assert np.allclose(ops.convert_to_numpy(out1_eval), ops.convert_to_numpy(out2_eval), atol=1e-5)

    norm1.reset_parameters()
    norm1.reset_running_stats()


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("mode", ["graph", "node"])
def test_layer_norm(affine, mode):
    x = random.normal((100, 16))
    batch = ops.zeros((100,), dtype="int32")

    norm = LayerNorm(16, affine=affine, mode=mode)

    out1 = norm(x)
    assert ops.shape(out1) == (100, 16)

    if mode == "graph":
        out1_b = norm(x, batch)
        assert np.allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out1_b), atol=1e-5)

        x2 = ops.concatenate([x, x], axis=0)
        batch2 = ops.concatenate([batch, batch + 1], axis=0)
        out2 = norm(x2, batch2)
        assert np.allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2[:100]), atol=1e-5)
        assert np.allclose(ops.convert_to_numpy(out1), ops.convert_to_numpy(out2[100:]), atol=1e-5)
    else:
        mean = ops.mean(out1, axis=-1)
        std = ops.std(out1, axis=-1)
        assert np.allclose(ops.convert_to_numpy(mean), 0.0, atol=1e-4)
        if not affine:
            assert np.allclose(ops.convert_to_numpy(std), 1.0, atol=1e-3)


@pytest.mark.parametrize("affine", [False, True])
def test_hetero_layer_norm(affine):
    x = random.normal((100, 16))
    expected = LayerNorm(16, affine=affine, mode="node")(x)

    type_vec = ops.zeros((100,), dtype="int32")
    type_ptr = [0, 100]

    norm = HeteroLayerNorm(16, num_types=1, affine=affine)

    out = norm(x, type_vec)
    assert ops.shape(out) == (100, 16)
    assert np.allclose(ops.convert_to_numpy(out), ops.convert_to_numpy(expected), atol=1e-3)

    out_ptr = norm(x, type_ptr=type_ptr)
    assert np.allclose(ops.convert_to_numpy(out_ptr), ops.convert_to_numpy(expected), atol=1e-3)


@pytest.mark.parametrize("conf", [True, False])
def test_batch_norm(conf):
    x = random.normal((100, 16))
    norm = BatchNorm(16, affine=conf, track_running_stats=conf)
    norm.reset_running_stats()
    norm.reset_parameters()

    assert norm.in_channels == 16
    assert norm.eps == 1e-5
    assert norm.momentum == 0.1
    assert norm.affine == conf
    assert norm.track_running_stats == conf
    assert (norm.weight is not None) == conf
    assert (norm.bias is not None) == conf

    out = norm(x)
    assert ops.shape(out) == (100, 16)


def test_batch_norm_single_element():
    x = random.normal((1, 16))

    norm = BatchNorm(16)
    with pytest.raises(ValueError, match="Expected more than 1 value"):
        norm(x)

    with pytest.raises(ValueError, match="requires 'track_running_stats'"):
        BatchNorm(16, track_running_stats=False, allow_single_element=True)

    norm = BatchNorm(16, track_running_stats=True, allow_single_element=True)
    out = norm(x)
    assert np.allclose(ops.convert_to_numpy(out), ops.convert_to_numpy(x), atol=1e-5)


@pytest.mark.parametrize("conf", [True, False])
def test_hetero_batch_norm(conf):
    x = random.normal((100, 16))

    norm = BatchNorm(16, affine=conf, track_running_stats=conf)
    expected = norm(x)

    type_vec = ops.zeros((100,), dtype="int32")
    h_norm = HeteroBatchNorm(16, num_types=1, affine=conf, track_running_stats=conf)
    h_norm.reset_running_stats()
    h_norm.reset_parameters()

    out = h_norm(x, type_vec)
    assert ops.shape(out) == (100, 16)
    assert np.allclose(ops.convert_to_numpy(out), ops.convert_to_numpy(expected), atol=1e-4)


def test_diff_group_norm():
    x = random.normal((100, 16))
    norm = DiffGroupNorm(16, groups=4, lamda=0.01)

    out = norm(x)
    assert ops.shape(out) == (100, 16)

    norm.reset_parameters()


def test_group_distance_ratio():
    x = random.normal((6, 16))
    y = ops.convert_to_tensor(np.array([0, 1, 0, 1, 1, 1], dtype=np.int64))

    assert DiffGroupNorm.group_distance_ratio(x, y) > 0
