"""
Training verification tests for Dense / Linear layers in k3_node.layers.dense.
"""

import pytest
from tests_training.common import DENSE_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", DENSE_LAYERS)
def test_dense_layer_training(layer_name):
    train_and_verify_layer(layer_name)

