"""
Training verification tests for Convolution layers in k3_node.layers.conv.
"""

import pytest
from tests_training.common import CONV_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", CONV_LAYERS)
def test_conv_layer_training(layer_name):
    train_and_verify_layer(layer_name)

