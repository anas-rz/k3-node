"""
Training verification tests for Knowledge Graph Embedding layers in k3_node.layers.kge.
"""

import pytest
from tests_training.common import KGE_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", KGE_LAYERS)
def test_kge_layer_training(layer_name):
    train_and_verify_layer(layer_name)

