"""Unit tests for K3-Node high-level Task APIs."""

import os
import tempfile
import pytest
import numpy as np
import keras
from keras import ops

import k3_node
from k3_node.data import Data
from k3_node.tasks import (
    BaseTask,
    resolve_backbone,
    NodeClassifier,
    NodeRegressor,
    GraphClassifier,
    GraphRegressor,
    LinkPredictor,
)


def _create_synthetic_node_data(num_nodes=24, in_channels=8, num_classes=3, multi_label=False):
    x = np.random.randn(num_nodes, in_channels).astype("float32")
    edges_src = np.arange(num_nodes - 1, dtype="int64")
    edges_dst = np.arange(1, num_nodes, dtype="int64")
    edge_index = np.stack([edges_src, edges_dst], axis=0)

    if multi_label:
        y = np.random.randint(0, 2, size=(num_nodes, num_classes)).astype("float32")
    else:
        y = np.random.randint(0, num_classes, size=(num_nodes,)).astype("int64")

    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[: num_nodes // 2] = True
    val_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[num_nodes // 2 : 3 * num_nodes // 4] = True
    test_mask = ~(train_mask | val_mask)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def _create_synthetic_graph_dataset(num_graphs=12, nodes_per_graph=6, in_channels=8, is_regression=False):
    dataset = []
    for i in range(num_graphs):
        x = np.random.randn(nodes_per_graph, in_channels).astype("float32")
        edges_src = np.arange(nodes_per_graph - 1, dtype="int64")
        edges_dst = np.arange(1, nodes_per_graph, dtype="int64")
        edge_index = np.stack([edges_src, edges_dst], axis=0)

        if is_regression:
            y = np.array([float(np.mean(x))], dtype="float32")
        else:
            y = np.array(i % 2, dtype="int64")

        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset


# ==============================================================================
# 1. Top-Level Imports & Base Task Tests
# ==============================================================================

def test_tasks_module_exports():
    """Verify tasks are exported at both k3_node.tasks and k3_node top level."""
    assert hasattr(k3_node, "tasks")
    assert hasattr(k3_node, "NodeClassifier")
    assert hasattr(k3_node, "NodeRegressor")
    assert hasattr(k3_node, "GraphClassifier")
    assert hasattr(k3_node, "GraphRegressor")
    assert hasattr(k3_node, "LinkPredictor")

    assert k3_node.NodeClassifier is NodeClassifier
    assert k3_node.NodeRegressor is NodeRegressor
    assert k3_node.GraphClassifier is GraphClassifier
    assert k3_node.GraphRegressor is GraphRegressor
    assert k3_node.LinkPredictor is LinkPredictor


def test_base_task():
    """Test BaseTask compile, extract_inputs, save and load."""
    dummy_model = keras.Sequential([keras.layers.Dense(4)])
    task = BaseTask(model=dummy_model)
    task.compile(optimizer="adam", loss="mse")
    assert task._is_compiled

    # extract_inputs
    data = _create_synthetic_node_data(num_nodes=5, in_channels=4)
    extracted = task._extract_inputs(data)
    assert isinstance(extracted, tuple)

    # save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_task.keras")
        task.save(filepath)
        loaded = BaseTask.load(filepath)
        assert loaded.model is not None


def test_resolve_backbone():
    """Test backbone resolver with valid strings, custom models, and errors."""
    for name in ["gcn", "gat", "sage", "gin", "mlp"]:
        model = resolve_backbone(name, in_channels=8, hidden_channels=16, out_channels=4, num_layers=2)
        assert isinstance(model, keras.Model)

    # Custom model pass-through
    custom_model = keras.Sequential([keras.layers.Dense(4)])
    resolved_custom = resolve_backbone(custom_model, in_channels=8, out_channels=4)
    assert resolved_custom is custom_model

    # Unknown backbone
    with pytest.raises(ValueError):
        resolve_backbone("non_existent_backbone", in_channels=8, out_channels=4)

    # Invalid type
    with pytest.raises(TypeError):
        resolve_backbone(12345, in_channels=8, out_channels=4)


# ==============================================================================
# 2. NodeClassifier Tests
# ==============================================================================

@pytest.mark.parametrize("backbone", ["gcn", "sage"])
def test_node_classifier_fit_predict_evaluate(backbone):
    data = _create_synthetic_node_data(num_nodes=20, in_channels=8, num_classes=3)
    clf = NodeClassifier(backbone=backbone, hidden_channels=16, num_layers=2, dropout=0.0)

    # Fit
    clf.fit(data, epochs=2, lr=0.01, verbose=0)
    assert clf.model is not None

    # Predict discrete labels
    preds = clf.predict(data, mask="test_mask")
    assert ops.shape(preds)[0] == int(ops.convert_to_numpy(ops.sum(ops.cast(data.test_mask, "int32"))))

    # Predict probabilities
    probs = clf.predict_proba(data, mask="test_mask")
    assert ops.shape(probs)[-1] == 3

    # Evaluate
    metrics = clf.evaluate(data, mask="test_mask")
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_node_classifier_multi_label():
    data = _create_synthetic_node_data(num_nodes=20, in_channels=8, num_classes=4, multi_label=True)
    clf = NodeClassifier(backbone="gcn", hidden_channels=16, num_layers=2, multi_label=True)
    clf.fit(data, epochs=2, verbose=0)

    probs = clf.predict_proba(data, mask="test_mask")
    assert ops.shape(probs)[-1] == 4

    preds = clf.predict(data, mask="test_mask")
    assert ops.shape(preds)[-1] == 4


# ==============================================================================
# 3. NodeRegressor Tests
# ==============================================================================

def test_node_regressor_fit_predict_evaluate():
    data = _create_synthetic_node_data(num_nodes=20, in_channels=8, num_classes=1)
    data.y = np.random.randn(20, 1).astype("float32")

    reg = NodeRegressor(backbone="gat", hidden_channels=16, num_layers=2)
    reg.fit(data, epochs=2, lr=0.01, verbose=0)

    preds = reg.predict(data, mask="test_mask")
    assert ops.shape(preds)[0] == int(ops.convert_to_numpy(ops.sum(ops.cast(data.test_mask, "int32"))))

    metrics = reg.evaluate(data, mask="test_mask")
    assert "mae" in metrics
    assert "mse" in metrics
    assert "loss" in metrics
    assert metrics["mae"] >= 0.0


# ==============================================================================
# 4. GraphClassifier Tests
# ==============================================================================

@pytest.mark.parametrize("pooling", ["mean", "add", "max"])
def test_graph_classifier_fit_predict_evaluate(pooling):
    dataset = _create_synthetic_graph_dataset(num_graphs=10, nodes_per_graph=5, in_channels=6)
    clf = GraphClassifier(
        backbone="gin",
        hidden_channels=16,
        num_layers=2,
        pooling=pooling,
        dropout=0.0,
    )

    clf.fit(dataset, epochs=2, batch_size=4, verbose=0)

    # Predict
    preds = clf.predict(dataset[:4], batch_size=4)
    assert ops.shape(preds)[0] == 4

    # Predict proba
    probs = clf.predict_proba(dataset[:4], batch_size=4)
    assert ops.shape(probs)[0] == 4
    assert ops.shape(probs)[1] == 2

    # Evaluate
    metrics = clf.evaluate(dataset[:4], batch_size=4)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ==============================================================================
# 5. GraphRegressor Tests
# ==============================================================================

@pytest.mark.parametrize("loss_name", ["mae", "mse"])
def test_graph_regressor_fit_predict_evaluate(loss_name):
    dataset = _create_synthetic_graph_dataset(num_graphs=10, nodes_per_graph=5, in_channels=6, is_regression=True)
    reg = GraphRegressor(
        backbone="sage",
        hidden_channels=16,
        num_layers=2,
        pooling="mean",
        loss=loss_name,
    )

    reg.fit(dataset, epochs=2, batch_size=4, verbose=0)

    preds = reg.predict(dataset[:4], batch_size=4)
    assert ops.shape(preds)[0] == 4

    metrics = reg.evaluate(dataset[:4], batch_size=4)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "loss" in metrics
    assert metrics["mae"] >= 0.0


# ==============================================================================
# 6. LinkPredictor Tests
# ==============================================================================

@pytest.mark.parametrize("decoder", ["inner_product", "cosine", "mlp"])
def test_link_predictor_decoders_and_fit(decoder):
    data = _create_synthetic_node_data(num_nodes=16, in_channels=8)
    lp = LinkPredictor(
        backbone="gcn",
        hidden_channels=16,
        out_channels=16,
        num_layers=2,
        decoder=decoder,
    )

    # Dynamic negative sampling training
    lp.fit(data, epochs=2, lr=0.01, neg_ratio=1.0, verbose=0)

    # Encode node embeddings
    z = lp.encode(data)
    assert ops.shape(z) == (16, 16)

    # Predict proba & binary labels
    query_edges = np.array([[0, 1, 2], [1, 2, 3]], dtype="int32")
    probs = lp.predict_proba(data, edge_label_index=query_edges)
    assert ops.shape(probs) == (3,)

    preds = lp.predict(data, edge_label_index=query_edges, threshold=0.5)
    assert ops.shape(preds) == (3,)

    # Evaluate
    metrics = lp.evaluate(data, edge_label_index=query_edges, edge_label=[1, 1, 0])
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_link_predictor_explicit_labels():
    data = _create_synthetic_node_data(num_nodes=16, in_channels=8)
    edge_label_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int32")
    edge_label = np.array([1, 1, 0, 0], dtype="float32")

    lp = LinkPredictor(backbone="gcn", hidden_channels=16, out_channels=16)
    lp.fit(data, edge_label_index=edge_label_index, edge_label=edge_label, epochs=2, verbose=0)

    metrics = lp.evaluate(data, edge_label_index=edge_label_index, edge_label=edge_label)
    assert "accuracy" in metrics
    assert "auc" in metrics
    assert "ap" in metrics


# ==============================================================================
# 7. Applications API Tests
# ==============================================================================

def test_applications_api_exports():
    """Verify k3_node.applications submodules and model exports."""
    import k3_node.applications as apps
    assert hasattr(apps, "chemistry")
    assert hasattr(apps, "materials")
    assert hasattr(apps, "bio")

    # Chemistry exports
    assert hasattr(apps.chemistry, "AttentiveFP")
    assert hasattr(apps.chemistry, "SchNet")
    assert hasattr(apps.chemistry, "DimeNetPlusPlus")

    # Materials exports
    assert hasattr(apps.materials, "MEGNet")
    assert hasattr(apps.materials, "M3GNet")
    assert hasattr(apps.materials, "CHGNet")

    # Bio exports
    assert hasattr(apps.bio, "UniMolDockingModel")
    assert hasattr(apps.bio, "DockingPoseModelV2")
