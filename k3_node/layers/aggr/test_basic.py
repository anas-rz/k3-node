import pytest
from keras import ops, random
from k3_node.layers import SumAggregation, MaxAggregation, MeanAggregation, SoftmaxAggregation, PowerMeanAggregation
@pytest.mark.parametrize('Aggregation', [
    MeanAggregation,
    SumAggregation,
    MaxAggregation,
])
def test_basic_aggregation(Aggregation):
    x = random.normal((6, 16))
    index = ops.convert_to_tensor([0, 0, 1, 1, 1, 2])

    aggr = Aggregation()

    out = aggr(x, index)
    assert ops.shape(out) == (3, ops.shape(x)[-1])