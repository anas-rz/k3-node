import pytest
from keras import ops, random, layers
from k3_node.layers import (
    SumAggregation,
    MaxAggregation,
    MeanAggregation,
    SoftmaxAggregation,
    PowerMeanAggregation,
    DeepSetsAggregation,
)


@pytest.mark.parametrize(
    "Aggregation",
    [
        MeanAggregation,
        SumAggregation,
        MaxAggregation,
    ],
)
def test_basic_aggregation(Aggregation):
    x = random.normal((6, 16))
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2])

    aggr = Aggregation()

    out = aggr(x, index)
    assert ops.shape(out) == (3, ops.shape(x)[-1])


@pytest.mark.parametrize(
    "in_channels, local_units, global_units",
    [
        (16, 32, 64),
        (8, 16, 32),
        (10, 20, 30),
    ],
)
def test_deep_sets_aggregation(in_channels, local_units, global_units):
    x = random.normal((6, in_channels))
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2])

    aggr = DeepSetsAggregation(
        local_mlp=layers.Dense(local_units),
        global_mlp=layers.Dense(global_units),
    )

    out = aggr(x, index)
    assert ops.shape(out) == (3, global_units)
