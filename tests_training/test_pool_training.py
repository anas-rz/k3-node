"""
Training verification tests for Pooling layers in k3_node.layers.pool.
"""

import pytest
from tests_training.common import POOL_LAYERS, train_and_verify_layer


@pytest.mark.parametrize("layer_name", POOL_LAYERS)
def test_pool_layer_training(layer_name):
    train_and_verify_layer(layer_name)



@pytest.mark.parametrize("layer_name", ["ClusterPooling", "EdgePooling"])
def test_host_side_pooling_symbolic_build(layer_name):
    """Keras shape inference must not run host-side clustering on placeholder values.

    On the torch backend this used to request tens of gigabytes (ClusterPooling asked for 44 GB).
    """
    from tests_training.common import EAGER_ONLY_LAYERS, get_layer_test

    model_factory, inputs_factory, target_factory, _, _ = get_layer_test(layer_name)
    model = model_factory()  # deliberately not built eagerly
    model.compile(optimizer="adam", loss="mse", run_eagerly=layer_name in EAGER_ONLY_LAYERS)
    model.train_on_batch(inputs_factory(), target_factory())
