"""
Training verification tests for Pooling layers in k3_node.layers.pool.
"""

import pytest
from tests_training.common import POOL_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", POOL_LAYERS)
def test_pool_layer_training(layer_name):
    train_and_verify_layer(layer_name)

