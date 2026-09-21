"""
Training verification tests for Attention layers in k3_node.layers.attention.
"""

import pytest
from tests_training.common import ATTENTION_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", ATTENTION_LAYERS)
def test_attention_layer_training(layer_name):
    train_and_verify_layer(layer_name)

