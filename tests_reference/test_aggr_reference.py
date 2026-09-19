import numpy as np
import torch
from keras import ops

import torch_geometric.nn.aggr as pyg_aggr
import k3_node.layers.aggr as k3_aggr


def test_reference_basic_aggregations():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    # Sum
    pyg_out = pyg_aggr.SumAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.SumAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Mean
    pyg_out = pyg_aggr.MeanAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.MeanAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Max
    pyg_out = pyg_aggr.MaxAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.MaxAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Min
    pyg_out = pyg_aggr.MinAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.MinAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Mul
    pyg_out = pyg_aggr.MulAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.MulAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Var
    pyg_out = pyg_aggr.VarAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.VarAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Std
    pyg_out = pyg_aggr.StdAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.StdAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)


def test_reference_softmax_aggregation():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    pyg_aggr_layer = pyg_aggr.SoftmaxAggregation(t=2.0)
    k3_aggr_layer = k3_aggr.SoftmaxAggregation(t=2.0)

    pyg_out = pyg_aggr_layer(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr_layer(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)


def test_reference_powermean_aggregation():
    x = np.random.uniform(0.1, 2.0, (6, 4)).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    pyg_aggr_layer = pyg_aggr.PowerMeanAggregation(p=2.0)
    k3_aggr_layer = k3_aggr.PowerMeanAggregation(p=2.0)

    pyg_out = pyg_aggr_layer(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr_layer(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-4)


def test_reference_quantile_aggregation():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    # Median
    pyg_out = pyg_aggr.MedianAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.MedianAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    # Quantile (linear)
    pyg_out = pyg_aggr.QuantileAggregation(q=0.5, interpolation="linear")(
        torch.from_numpy(x), torch.from_numpy(index)
    )
    k3_out = k3_aggr.QuantileAggregation(q=0.5, interpolation="linear")(
        ops.convert_to_tensor(x), ops.convert_to_tensor(index)
    )
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)


def test_reference_variance_preserving_aggregation():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    pyg_out = pyg_aggr.VariancePreservingAggregation()(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.VariancePreservingAggregation()(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)


def test_reference_scaler_aggregation():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
    deg = torch.tensor([0, 1, 2, 3], dtype=torch.float)

    pyg_aggr_layer = pyg_aggr.DegreeScalerAggregation(
        aggr=["mean"],
        scaler=["identity", "amplification", "linear"],
        deg=deg,
    )
    k3_aggr_layer = k3_aggr.DegreeScalerAggregation(
        aggr=["mean"],
        scaler=["identity", "amplification", "linear"],
        deg=ops.convert_to_tensor(deg.numpy()),
    )

    pyg_out = pyg_aggr_layer(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr_layer(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)


def test_reference_multi_aggregation():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    pyg_aggr_layer = pyg_aggr.MultiAggregation(["sum", "mean", "max"], mode="cat")
    k3_aggr_layer = k3_aggr.MultiAggregation(["sum", "mean", "max"], mode="cat")

    pyg_out = pyg_aggr_layer(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr_layer(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)

    pyg_sum_mode = pyg_aggr.MultiAggregation(["sum", "mean"], mode="sum")(
        torch.from_numpy(x), torch.from_numpy(index)
    )
    k3_sum_mode = k3_aggr.MultiAggregation(["sum", "mean"], mode="sum")(
        ops.convert_to_tensor(x), ops.convert_to_tensor(index)
    )
    assert np.allclose(pyg_sum_mode.numpy(), ops.convert_to_numpy(k3_sum_mode), atol=1e-5)


def test_reference_sort_aggregation():
    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    pyg_out = pyg_aggr.SortAggregation(k=2)(torch.from_numpy(x), torch.from_numpy(index))
    k3_out = k3_aggr.SortAggregation(k=2)(ops.convert_to_tensor(x), ops.convert_to_tensor(index))
    assert np.allclose(pyg_out.numpy(), ops.convert_to_numpy(k3_out), atol=1e-5)


def test_reference_deep_sets_and_attentional():
    from keras import layers

    x = np.random.randn(6, 4).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    # DeepSets
    pyg_ds = pyg_aggr.DeepSetsAggregation(local_nn=torch.nn.Identity(), global_nn=torch.nn.Identity())
    k3_ds = k3_aggr.DeepSetsAggregation(local_nn=lambda t: t, global_nn=lambda t: t)
    assert np.allclose(pyg_ds(torch.from_numpy(x), torch.from_numpy(index)).numpy(),
                       ops.convert_to_numpy(k3_ds(ops.convert_to_tensor(x), ops.convert_to_tensor(index))), atol=1e-5)

    # Attentional
    torch.manual_seed(42)
    pyg_gate = torch.nn.Linear(4, 1)
    pyg_att = pyg_aggr.AttentionalAggregation(gate_nn=pyg_gate)
    k3_gate = layers.Dense(1)
    k3_att = k3_aggr.AttentionalAggregation(gate_nn=k3_gate)
    k3_gate.build((None, 4))
    with torch.no_grad():
        k3_gate.set_weights([pyg_gate.weight.detach().numpy().T, pyg_gate.bias.detach().numpy()])

    assert np.allclose(pyg_att(torch.from_numpy(x), torch.from_numpy(index)).detach().numpy(),
                       ops.convert_to_numpy(k3_att(ops.convert_to_tensor(x), ops.convert_to_tensor(index))), atol=1e-5)

