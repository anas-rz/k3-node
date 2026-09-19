import numpy as np
import pytest
from keras import layers, ops

from k3_node.layers.aggr import (
    Aggregation,
    AttentionalAggregation,
    DeepSetsAggregation,
    DegreeScalerAggregation,
    EquilibriumAggregation,
    FusedAggregation,
    GRUAggregation,
    GraphMultisetTransformer,
    LCMAggregation,
    LSTMAggregation,
    MLPAggregation,
    MaxAggregation,
    MeanAggregation,
    MedianAggregation,
    MinAggregation,
    MulAggregation,
    MultiAggregation,
    PatchTransformerAggregation,
    PowerMeanAggregation,
    QuantileAggregation,
    Set2Set,
    SetTransformerAggregation,
    SoftmaxAggregation,
    SortAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
    VariancePreservingAggregation,
    aggregation_resolver,
    ptr2index,
    to_dense_batch,
)


def test_ptr2index_and_to_dense_batch():
    ptr = ops.convert_to_tensor([0, 2, 5, 6], dtype="int32")
    index = ptr2index(ptr)
    assert np.array_equal(ops.convert_to_numpy(index), [0, 0, 1, 1, 1, 2])

    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")

    dense_x, mask = to_dense_batch(x, index)
    assert ops.shape(dense_x) == (3, 3, 2)
    assert ops.shape(mask) == (3, 3)
    assert np.array_equal(ops.convert_to_numpy(mask), [
        [True, True, False],
        [True, True, True],
        [True, False, False],
    ])


@pytest.mark.parametrize(
    "AggrClass,expected",
    [
        (SumAggregation, [[4.0, 6.0], [21.0, 24.0], [11.0, 12.0]]),
        (MeanAggregation, [[2.0, 3.0], [7.0, 8.0], [11.0, 12.0]]),
        (MaxAggregation, [[3.0, 4.0], [9.0, 10.0], [11.0, 12.0]]),
        (MinAggregation, [[1.0, 2.0], [5.0, 6.0], [11.0, 12.0]]),
        (MulAggregation, [[3.0, 8.0], [315.0, 480.0], [11.0, 12.0]]),
    ],
)
def test_basic_reductions(AggrClass, expected):
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    aggr = AggrClass()
    out = aggr(x, index)
    assert np.allclose(ops.convert_to_numpy(out), expected)

    # Test with ptr
    ptr = ops.convert_to_tensor([0, 2, 5, 6], dtype="int32")
    out_ptr = aggr(x, ptr=ptr)
    assert np.allclose(ops.convert_to_numpy(out_ptr), expected)


def test_var_and_std_aggregation():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    var_aggr = VarAggregation()
    var_out = var_aggr(x, index)
    assert ops.shape(var_out) == (3, 2)
    assert np.all(ops.convert_to_numpy(var_out) >= 0.0)

    std_aggr = StdAggregation()
    std_out = std_aggr(x, index)
    assert ops.shape(std_out) == (3, 2)
    assert np.all(ops.convert_to_numpy(std_out) >= 0.0)


def test_softmax_aggregation():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    # Fixed temperature
    aggr = SoftmaxAggregation(t=1.0)
    out = aggr(x, index)
    assert ops.shape(out) == (3, 2)

    # Learnable multi-channel temperature
    learnable_aggr = SoftmaxAggregation(t=1.0, learn=True, channels=2)
    out_learn = learnable_aggr(x, index)
    assert ops.shape(out_learn) == (3, 2)


def test_powermean_aggregation():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    # Fixed p
    aggr = PowerMeanAggregation(p=2.0)
    out = aggr(x, index)
    assert ops.shape(out) == (3, 2)

    # Learnable p
    learnable_aggr = PowerMeanAggregation(p=1.0, learn=True, channels=2)
    out_learn = learnable_aggr(x, index)
    assert ops.shape(out_learn) == (3, 2)


def test_quantile_and_median_aggregation():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    # Median
    med_aggr = MedianAggregation()
    med_out = med_aggr(x, index)
    assert ops.shape(med_out) == (3, 2)

    # Quantile with multiple q
    q_aggr = QuantileAggregation(q=[0.25, 0.75], interpolation="linear")
    q_out = q_aggr(x, index)
    assert ops.shape(q_out) == (3, 4)

    # Interpolations
    for interp in ["lower", "higher", "nearest", "midpoint"]:
        q_interp = QuantileAggregation(q=0.5, interpolation=interp)
        res = q_interp(x, index)
        assert ops.shape(res) == (3, 2)


def test_attentional_aggregation():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    gate_nn = layers.Dense(1)
    nn = layers.Dense(8)
    aggr = AttentionalAggregation(gate_nn=gate_nn, nn=nn)

    out = aggr(x, index)
    assert ops.shape(out) == (3, 8)


def test_set2set_aggregation():
    in_channels = 4
    x = ops.ones((6, in_channels), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    s2s = Set2Set(in_channels=in_channels, processing_steps=2)
    out = s2s(x, index)
    assert ops.shape(out) == (3, 2 * in_channels)


def test_degree_scaler_aggregation():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")
    deg = ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0], dtype="float32")

    scalers = ["identity", "amplification", "attenuation", "linear", "inverse_linear"]
    aggr = DegreeScalerAggregation(aggr="mean", scaler=scalers, deg=deg)

    out = aggr(x, index)
    assert ops.shape(out) == (3, 4 * len(scalers))


def test_sort_aggregation():
    x = ops.convert_to_tensor([
        [1.0, 2.0],
        [3.0, 1.0],
        [5.0, 9.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 5.0],
    ], dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    aggr = SortAggregation(k=2)
    out = aggr(x, index)
    assert ops.shape(out) == (3, 2 * 2)


def test_multi_aggregation():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    # Cat mode
    aggr_cat = MultiAggregation(["mean", "max"], mode="cat")
    assert ops.shape(aggr_cat(x, index)) == (3, 8)

    # Proj mode
    aggr_proj = MultiAggregation(["mean", "max"], mode="proj", mode_kwargs={"in_channels": 4, "out_channels": 16})
    assert ops.shape(aggr_proj(x, index)) == (3, 16)

    # Elementwise modes
    for m in ["sum", "mean", "max", "min", "var", "std"]:
        aggr_m = MultiAggregation(["sum", "mean"], mode=m)
        assert ops.shape(aggr_m(x, index)) == (3, 4)


def test_deep_sets_and_mlp_aggregation():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    # DeepSets
    ds = DeepSetsAggregation(local_nn=layers.Dense(8), global_nn=layers.Dense(16))
    assert ops.shape(ds(x, index)) == (3, 16)

    # MLP
    mlp_aggr = MLPAggregation(in_channels=4, out_channels=16, max_num_elements=3)
    assert ops.shape(mlp_aggr(x, index)) == (3, 16)


def test_lstm_and_gru_aggregation():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    lstm_aggr = LSTMAggregation(in_channels=4, out_channels=8)
    assert ops.shape(lstm_aggr(x, index)) == (3, 8)

    gru_aggr = GRUAggregation(in_channels=4, out_channels=8)
    assert ops.shape(gru_aggr(x, index)) == (3, 8)


def test_set_transformer_and_gmt():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    st = SetTransformerAggregation(channels=4, num_seed_points=2, heads=1, concat=True)
    out_st = st(x, index)
    assert ops.shape(out_st) == (3, 8)

    gmt = GraphMultisetTransformer(channels=4, k=2, heads=1)
    out_gmt = gmt(x, index)
    assert ops.shape(out_gmt) == (3, 4)


def test_variance_preserving_and_patch_transformer():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    vpa = VariancePreservingAggregation()
    assert ops.shape(vpa(x, index)) == (3, 4)

    pt = PatchTransformerAggregation(in_channels=4, out_channels=8, patch_size=2, hidden_channels=4)
    assert ops.shape(pt(x, index)) == (3, 8)


def test_lcm_and_equilibrium_aggregation():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    lcm = LCMAggregation(in_channels=4, out_channels=4, project=False)
    assert ops.shape(lcm(x, index)) == (3, 4)

    eq = EquilibriumAggregation(in_channels=4, out_channels=4, num_layers=[8])
    assert ops.shape(eq(x, index)) == (3, 4)


def test_fused_and_resolver():
    x = ops.ones((6, 4), dtype="float32")
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2], dtype="int32")

    fused = FusedAggregation(["sum", "mean", "max"])
    outs = fused(x, index)
    assert len(outs) == 3
    for o in outs:
        assert ops.shape(o) == (3, 4)

    # Resolver
    res_sum = aggregation_resolver("sum")
    assert isinstance(res_sum, SumAggregation)
    res_mean = aggregation_resolver("mean")
    assert isinstance(res_mean, MeanAggregation)
