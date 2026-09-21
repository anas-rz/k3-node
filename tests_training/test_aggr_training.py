"""
Training verification tests for Aggregation layers in k3_node.layers.aggr.
"""

import pytest
from tests_training.common import AGGR_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", AGGR_LAYERS)
def test_aggr_layer_training(layer_name):
    train_and_verify_layer(layer_name)

