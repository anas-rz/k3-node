"""
Training verification tests for Normalization layers in k3_node.layers.norm.
"""

import pytest
from tests_training.common import NORM_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", NORM_LAYERS)
def test_norm_layer_training(layer_name):
    train_and_verify_layer(layer_name)

